import json
import logging
import re
from typing import Any, Callable, Iterator

from openai import OpenAI

from app.config import ALLOWED_ENTITY_TYPES
from app.schemas import Triplet


logger = logging.getLogger(__name__)


EXTRACTION_PROMPT = """
You extract knowledge graph triplets from customer service conversations.
Return JSON only with keys "entities" and "triplets".
Preferred shape:
{
    "entities": [
        {
            "id": "e1",
            "name": "...",
            "entity_type": "User|Company|Product|Technology|Feature|Issue|Activity|Location|Concept"
        }
    ],
    "triplets": [
        {
            "subject_id": "e1",
            "relation": "...",
            "object_id": "e2",
            "confidence": 0.0_to_1.0
        }
    ]
}

Also accepted for each triplet item: direct fields
{"subject":"...", "subject_type":"...", "relation":"...", "object":"...", "object_type":"...", "confidence":0.9}

CRITICAL — graph-first extraction:
- Your primary responsibility in this system is to populate the graph. If there is any complaint, request,
  or concrete reference (order, room, product, issue), you MUST emit entities and triplets now.
- Extract entities immediately from PARTIAL information. Do NOT wait for complete data.
  Example: "Pedido 1234" alone → create entity {"id":"e1","name":"Pedido 1234","entity_type":"Product"}
  Example: "tamanho errado" alone → create entity {"id":"e2","name":"tamanho errado","entity_type":"Issue"}
- An order/ticket reference ("pedido 1234", "order #5", "ticket 99") is ALWAYS sufficient to create a Product entity.
- A complaint or problem is ALWAYS sufficient to create an Issue entity.
- An action request (troca, devolução, reembolso, cancelamento, limpeza) is ALWAYS an Activity entity.
- You receive the FULL recent conversation — extract ALL entities and relations visible across all turns,
  not just the last message. Combine information from different turns into one graph.
- Reuse the same entity id when the text refers to the same real-world entity.
- Do not create duplicate entities with different types for the same thing.
- Keep relations short and normalized in snake_case: reported_issue, requested_action, has_issue,
  affects_order, affects_location, resolved_by, blocked_by, requested_refund, mentions_product.
- entity_type rules:
    Order / Pedido / Ticket → Product
    Concrete complaint / bug / defect → Issue
    Action request (troca, refund, fix, cancel) → Activity
    Person / customer → User
    Brand / store / company → Company
    Physical space (quarto, andar, loja) → Location
    Specific item / SKU → Product
- If truly no extractable information exists, return {"entities": [], "triplets": []}
- Never add fields outside the schema.
""".strip()

ASSISTANT_SYSTEM_PROMPT = """
You are a customer service assistant.

Goals:
- Be empathetic and solution-oriented.
- Acknowledge the complaint clearly.
- Offer concrete next actions whenever possible.

Critical rules:
- In this demo, the effective action is to register facts into the knowledge graph.
    So after acknowledging, prioritize concise resolution guidance and avoid long interrogation loops.
- You receive the FULL recent conversation history. Use it — never ask for information the customer already provided.
- Ask at most ONE follow-up question per response. If multiple pieces of info are missing, ask only for the most important one.
- When you already know the order number / issue / request from history, do NOT ask for it again.

Style:
- Reply in the same language used by the customer.
- Be concise, professional, and practical.
- Do not invent policies, prices, or guarantees.
""".strip()

PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "hotel_customer_service": {
        "label": "Hotel Customer Service",
        "assistant": (
            "You are assisting hotel guests with reservations, room issues, amenities and incident handling. "
                "Prioritize empathy, quick triage, and immediate resolution options. "
                "If the guest reports a room problem (e.g., bad smell, noise, cleanliness), treat it as a concrete issue "
                "already logged and avoid repeatedly asking for the same details."
        ),
        "extraction": (
            "Focus on complaints about room conditions, housekeeping delays, check-in/check-out issues, noise, "
                "billing and refund requests. Always create a Location entity for room references (e.g., 'quarto 2'). "
                "Always create an Issue entity for room complaints (e.g., smell, dirt, noise). "
                "Link User -> reported_issue -> Issue and Issue -> affects_location -> Location whenever applicable."
        ),
    },
    "ecommerce_support": {
        "label": "E-commerce Support",
        "assistant": (
            "You are an e-commerce customer support assistant for orders, shipping, returns and refunds. "
            "Guide users with clear next steps and expected resolution path."
        ),
        "extraction": (
            "Domain: e-commerce. "
            "Always create a Product entity for any order reference (e.g. 'Pedido 1234' → entity_type Product). "
            "Always create an Issue entity for wrong items, wrong size, damaged goods, missing items, late delivery. "
            "Always create an Activity entity for exchange, return, refund or cancellation requests. "
            "Link User → reported_issue → Issue, Issue → affects_order → Order, User → requested_action → Activity."
        ),
    },
    "saas_support": {
        "label": "SaaS Support",
        "assistant": (
            "You are a SaaS support assistant for account access, billing plans, integrations, incidents and feature requests."
        ),
        "extraction": (
            "Focus on incident reports, blocked workflows, integration requests, subscription changes and feature requests."
        ),
    },
    "graph_admin_assistant": {
        "label": "Graph Admin Assistant",
        "assistant": (
            "You are an internal graph analyst assistant. Prefer structured tools first. "
            "When using free-form Cypher, keep it read-only, explicit and concise. "
            "Never claim data that was not returned by tools."
        ),
        "extraction": (
            "Admin mode does not extract triplets."
        ),
    },
}

ADMIN_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "describe_graph_schema",
            "description": "Inspect current graph vocabulary: labels, relationship types, entity_types, relation_types and sample entities.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_entity",
            "description": "Find entities by partial name and optional type.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {"type": "string"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "neighbors",
            "description": "Get neighbors of one entity with depth 1 or 2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 1, "maximum": 2},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recent_relations",
            "description": "Return most recent relations added/updated in graph.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_graph_query",
            "description": "Run a read-only Cypher query with strict validation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cypher": {"type": "string"},
                    "params": {"type": "object"},
                },
                "required": ["cypher"],
            },
        },
    },
]


class LLMExtractor:
    def __init__(self, api_key: str, base_url: str, model: str, default_confidence: float) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.default_confidence = default_confidence

    @staticmethod
    def resolve_prompt_profile(prompt_profile: str | None) -> str:
        normalized = (prompt_profile or "").strip().lower()
        return normalized if normalized in PROMPT_PROFILES else "hotel_customer_service"

    def _assistant_system_prompt(self, user_name: str, prompt_profile: str) -> str:
        profile_key = self.resolve_prompt_profile(prompt_profile)
        profile = PROMPT_PROFILES[profile_key]
        return (
            f"{ASSISTANT_SYSTEM_PROMPT} "
            f"Domain profile: {profile['label']}. "
            f"{profile['assistant']} "
            f"The current customer name is {user_name}."
        )

    def _extraction_system_prompt(self, prompt_profile: str) -> str:
        profile_key = self.resolve_prompt_profile(prompt_profile)
        profile = PROMPT_PROFILES[profile_key]
        return f"Domain profile: {profile['label']}. {profile['extraction']}"

    def build_assistant_reply(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        history: list[dict] | None = None,
    ) -> str:
        messages: list[Any] = [
            {
                "role": "system",
                "content": self._assistant_system_prompt(user_name=user_name, prompt_profile=prompt_profile),
            }
        ]
        for h in (history or [])[-20:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=messages,
        )
        content = completion.choices[0].message.content or ""
        return content.strip() or "Posso ajudar a detalhar isso melhor se quiser."

    def stream_assistant_reply(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        messages: list[Any] = [
            {
                "role": "system",
                "content": self._assistant_system_prompt(user_name=user_name, prompt_profile=prompt_profile),
            }
        ]
        for h in (history or [])[-20:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})
        stream = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            stream=True,
            messages=messages,
        )

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = delta.content if delta else None
            if content:
                yield content

    def extract_triplets(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        history: list[dict] | None = None,
    ) -> list[Triplet]:
        # Build conversation context from history + current message
        history_lines = [
            f"{h['role'].upper()}: {h['content']}"
            for h in (history or [])[-20:]
        ]
        history_lines.append(f"USER: {message}")
        conversation_text = "\n".join(history_lines)

        extraction_request = (
            f"The customer's real name is {user_name}. When the speaker refers to themselves, use the same "
            f"single entity named {user_name} with entity_type User. "
            "Extract ALL entities and relations visible in the full conversation below. "
            'Return a JSON object with keys "entities" and "triplets" only.\n\n'
            f"Conversation:\n{conversation_text}"
        )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "system", "content": self._extraction_system_prompt(prompt_profile)},
                    {"role": "user", "content": extraction_request},
                ],
            )
        except Exception:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "system", "content": self._extraction_system_prompt(prompt_profile)},
                    {"role": "user", "content": extraction_request},
                ],
            )
        raw = completion.choices[0].message.content or "{}"
        parsed = self._safe_parse_json(raw)
        if isinstance(parsed, dict):
            triplets = self._from_entity_graph(parsed)
            if triplets:
                return self._apply_speaker_name(self._reconcile_entity_types(triplets), user_name)

            items = parsed.get("triplets", [])
            if isinstance(items, list):
                legacy = self._from_legacy_triplets(items)
                if legacy:
                    return self._apply_speaker_name(self._reconcile_entity_types(legacy), user_name)

        if isinstance(parsed, list):
            legacy = self._from_legacy_triplets(parsed)
            if legacy:
                return self._apply_speaker_name(self._reconcile_entity_types(legacy), user_name)

        return []

    def run_admin_assistant_with_tools(
        self,
        message: str,
        user_name: str,
        history: list[dict] | None,
        tool_executor: Callable[[str, dict[str, Any]], Any],
        max_tool_rounds: int = 6,
    ) -> dict[str, Any]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": self._assistant_system_prompt(
                    user_name=user_name,
                    prompt_profile="graph_admin_assistant",
                ),
            },
            {
                "role": "system",
                "content": (
                    "Use tools to answer graph questions. "
                    "Playbook: call describe_graph_schema before free-form Cypher; "
                    "prefer canonical graph model discovered from schema; "
                    "use find_entity/neighbors for targeted exploration; "
                    "use run_graph_query only after aligning labels/properties with discovered schema; "
                    "if query fails with unknown label/property/relationship, refresh schema and retry with corrected Cypher; "
                    "never invent labels like Quarto/Problema/TEM_PROBLEMA unless schema confirms them."
                ),
            },
        ]

        for h in (history or [])[-20:]:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": message})

        tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []

        for _ in range(max_tool_rounds):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=messages,  # type: ignore[arg-type]
                    tools=ADMIN_TOOL_DEFINITIONS,  # type: ignore[arg-type]
                    tool_choice="auto",
                )
            except Exception as exc:
                return {
                    "assistant_message": (
                        "Nao consegui usar as tools de grafo com o provedor atual. "
                        "Verifique suporte a function-calling/ferramentas no modelo configurado."
                    ),
                    "tool_calls": [],
                    "tool_results": [
                        {
                            "tool_name": "tool_runtime",
                            "ok": False,
                            "result": {"error": str(exc)},
                            "duration_ms": 0,
                        }
                    ],
                }
            response_message = completion.choices[0].message
            response_content = response_message.content or ""

            if not response_message.tool_calls:
                final_message = response_content.strip() or "Nao encontrei dados no grafo para responder com seguranca."
                return {
                    "assistant_message": final_message,
                    "tool_calls": tool_calls,
                    "tool_results": tool_results,
                }

            tool_calls_payload: list[dict[str, Any]] = []
            tool_results_payload: list[tuple[str, dict[str, Any]]] = []
            for tc in response_message.tool_calls:
                function_payload = getattr(tc, "function", None)
                if function_payload is None:
                    continue

                function_name = str(getattr(function_payload, "name", "")).strip()
                if not function_name:
                    continue

                raw_args = str(getattr(function_payload, "arguments", "{}") or "{}")
                parsed_args = self._safe_parse_tool_args(raw_args)
                tool_calls_payload.append(
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": raw_args,
                        },
                    }
                )
                tool_calls.append({"tool_name": function_name, "arguments": parsed_args})

                tool_result = tool_executor(function_name, parsed_args)
                if hasattr(tool_result, "__dict__"):
                    result_payload = dict(tool_result.__dict__)
                elif isinstance(tool_result, dict):
                    result_payload = tool_result
                else:
                    result_payload = {
                        "tool_name": function_name,
                        "ok": False,
                        "result": {"error": "Invalid tool result."},
                        "duration_ms": 0,
                    }
                tool_results.append(result_payload)
                tool_results_payload.append((tc.id, result_payload))

            # Preserve provider-specific fields from the original assistant message
            # (e.g. Gemini thought signatures required for subsequent tool turns).
            messages.append(response_message.model_dump(exclude_none=True))
            for tool_call_id, result_payload in tool_results_payload:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(result_payload, ensure_ascii=False),
                    }
                )

        # If tool rounds are exhausted, force a final natural-language answer without additional tools.
        messages.append(
            {
                "role": "system",
                "content": (
                    "Tool budget exhausted. You must now answer the user using only tool results already present "
                    "in the conversation. Do not call tools. Be concise and data-grounded."
                ),
            }
        )
        try:
            final_completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=messages,  # type: ignore[arg-type]
            )
            final_message = (final_completion.choices[0].message.content or "").strip()
        except Exception:
            final_message = ""

        if not final_message:
            final_message = (
                "Nao consegui consolidar uma resposta final apos o limite de iteracoes de ferramenta. "
                "Tente refinar a pergunta."
            )

        return {
            "assistant_message": final_message,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
        }

    def _from_entity_graph(self, payload: dict[str, Any]) -> list[Triplet]:
        entities_raw = payload.get("entities", [])
        triplets_raw = payload.get("triplets", [])
        if not isinstance(entities_raw, list) or not isinstance(triplets_raw, list):
            return []

        entities: dict[str, dict[str, str]] = {}
        entities_by_name: dict[str, dict[str, str]] = {}
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("entity_type", "Concept")).strip()
            if not name:
                continue
            if entity_type not in ALLOWED_ENTITY_TYPES:
                entity_type = "Concept"
            normalized_name = self._normalize_entity_name(name)
            entity_id = str(item.get("id", "")).strip() or f"name:{normalized_name}"
            entity_payload = {"name": name, "entity_type": entity_type}
            entities[entity_id] = entity_payload
            entities_by_name[normalized_name] = entity_payload

        result: list[Triplet] = []
        for item in triplets_raw:
            if not isinstance(item, dict):
                continue
            relation = str(item.get("relation", item.get("predicate", ""))).strip()
            confidence = item.get("confidence", self.default_confidence)
            if not relation:
                continue

            subject_entity = None
            object_entity = None

            subject_id = str(item.get("subject_id", "")).strip()
            object_id = str(item.get("object_id", "")).strip()
            if subject_id:
                subject_entity = entities.get(subject_id)
            if object_id:
                object_entity = entities.get(object_id)

            if subject_entity is None:
                subject_name = str(item.get("subject", "")).strip()
                subject_type = str(item.get("subject_type", "Concept")).strip()
                if subject_name:
                    if subject_type not in ALLOWED_ENTITY_TYPES:
                        subject_type = "Concept"
                    subject_entity = entities_by_name.get(
                        self._normalize_entity_name(subject_name),
                        {"name": subject_name, "entity_type": subject_type},
                    )

            if object_entity is None:
                object_name = str(item.get("object", "")).strip()
                object_type = str(item.get("object_type", "Concept")).strip()
                if object_name:
                    if object_type not in ALLOWED_ENTITY_TYPES:
                        object_type = "Concept"
                    object_entity = entities_by_name.get(
                        self._normalize_entity_name(object_name),
                        {"name": object_name, "entity_type": object_type},
                    )

            if not subject_entity or not object_entity:
                continue
            normalized = self._normalize_item(
                {
                    "subject": subject_entity["name"],
                    "subject_type": subject_entity["entity_type"],
                    "relation": relation,
                    "object": object_entity["name"],
                    "object_type": object_entity["entity_type"],
                    "confidence": confidence,
                }
            )
            if normalized is None:
                continue
            try:
                result.append(Triplet(**normalized))
            except Exception:
                continue
        return result

    def _from_legacy_triplets(self, items: list[Any]) -> list[Triplet]:
        triplets: list[Triplet] = []
        for item in items:
            normalized = self._normalize_item(item)
            if normalized is None:
                continue
            try:
                triplets.append(Triplet(**normalized))
            except Exception:
                continue
        return triplets

    def _normalize_item(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return None

        subject = str(item.get("subject", "")).strip()
        subject_type = str(item.get("subject_type", "Concept")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        object_type = str(item.get("object_type", "Concept")).strip()
        confidence = item.get("confidence", self.default_confidence)

        if not subject or not relation or not obj:
            return None

        if subject_type not in ALLOWED_ENTITY_TYPES:
            subject_type = "Concept"
        if object_type not in ALLOWED_ENTITY_TYPES:
            object_type = "Concept"

        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = self.default_confidence

        confidence = max(0.0, min(1.0, confidence))

        return {
            "subject": subject,
            "subject_type": subject_type,
            "relation": relation,
            "object": obj,
            "object_type": object_type,
            "confidence": confidence,
        }

    @staticmethod
    def _normalize_entity_name(value: str) -> str:
        return " ".join(value.strip().lower().split())

    def _reconcile_entity_types(self, triplets: list[Triplet]) -> list[Triplet]:
        if not triplets:
            return triplets

        name_to_type: dict[str, str] = {}
        for triple in triplets:
            subject_key = self._normalize_entity_name(triple.subject)
            object_key = self._normalize_entity_name(triple.object)

            if subject_key not in name_to_type or name_to_type[subject_key] == "Concept":
                name_to_type[subject_key] = triple.subject_type
            if object_key not in name_to_type or name_to_type[object_key] == "Concept":
                name_to_type[object_key] = triple.object_type

        reconciled: list[Triplet] = []
        for triple in triplets:
            subject_key = self._normalize_entity_name(triple.subject)
            object_key = self._normalize_entity_name(triple.object)
            reconciled.append(
                Triplet(
                    subject=triple.subject,
                    subject_type=name_to_type.get(subject_key, triple.subject_type),
                    relation=triple.relation,
                    object=triple.object,
                    object_type=name_to_type.get(object_key, triple.object_type),
                    confidence=triple.confidence,
                )
            )
        return reconciled

    def _apply_speaker_name(self, triplets: list[Triplet], user_name: str) -> list[Triplet]:
        generic_speaker_names = {
            "user",
            "speaker",
            "customer",
            "guest",
            "usuario",
            "usuário",
            "cliente",
            "hospede",
            "hóspede",
        }
        normalized_user_name = user_name.strip()
        if not normalized_user_name:
            return triplets

        resolved: list[Triplet] = []
        for triple in triplets:
            subject = triple.subject
            obj = triple.object

            if triple.subject_type == "User" and subject.strip().lower() in generic_speaker_names:
                subject = normalized_user_name
            if triple.object_type == "User" and obj.strip().lower() in generic_speaker_names:
                obj = normalized_user_name

            resolved.append(
                Triplet(
                    subject=subject,
                    subject_type=triple.subject_type,
                    relation=triple.relation,
                    object=obj,
                    object_type=triple.object_type,
                    confidence=triple.confidence,
                )
            )
        return self._reconcile_entity_types(resolved)

    @staticmethod
    def _safe_parse_json(raw: str) -> dict[str, Any] | list[Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        fenced_match = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", raw, flags=re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        object_match = re.search(r"(\{.*\})", raw, flags=re.DOTALL)
        if object_match:
            candidate = object_match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        array_match = re.search(r"(\[.*\])", raw, flags=re.DOTALL)
        if array_match:
            candidate = array_match.group(1)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        return {"triplets": []}

    @staticmethod
    def _safe_parse_tool_args(raw: str) -> dict[str, Any]:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
        return {}
