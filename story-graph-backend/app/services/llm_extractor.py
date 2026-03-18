import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import ALLOWED_ENTITY_TYPES
from app.schemas import Triplet
from app.services.admin_graph_tools import AdminGraphTools, ToolExecution


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

CRITICAL - graph-first extraction:
- Your primary responsibility in this system is to populate the graph. If there is any complaint, request,
  or concrete reference (order, room, product, issue), you MUST emit entities and triplets now.
- Extract entities immediately from PARTIAL information. Do NOT wait for complete data.
  Example: "Pedido 1234" alone -> create entity {"id":"e1","name":"Pedido 1234","entity_type":"Product"}
  Example: "tamanho errado" alone -> create entity {"id":"e2","name":"tamanho errado","entity_type":"Issue"}
- An order/ticket reference ("pedido 1234", "order #5", "ticket 99") is ALWAYS sufficient to create a Product entity.
- A complaint or problem is ALWAYS sufficient to create an Issue entity.
- An action request (troca, devolucao, reembolso, cancelamento, limpeza) is ALWAYS an Activity entity.
- You receive the FULL recent conversation - extract ALL entities and relations visible across all turns,
  not just the last message. Combine information from different turns into one graph.
- Reuse the same entity id when the text refers to the same real-world entity.
- Do not create duplicate entities with different types for the same thing.
- Keep relations short and normalized in snake_case: reported_issue, requested_action, has_issue,
  affects_order, affects_location, resolved_by, blocked_by, requested_refund, mentions_product.
- entity_type rules:
    Order / Pedido / Ticket -> Product
    Concrete complaint / bug / defect -> Issue
    Action request (troca, refund, fix, cancel) -> Activity
    Person / customer -> User
    Brand / store / company -> Company
    Physical space (quarto, andar, loja) -> Location
    Specific item / SKU -> Product
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
- You receive the FULL recent conversation history. Use it - never ask for information the customer already provided.
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
            "Always create a Product entity for any order reference (e.g. 'Pedido 1234' -> entity_type Product). "
            "Always create an Issue entity for wrong items, wrong size, damaged goods, missing items, late delivery. "
            "Always create an Activity entity for exchange, return, refund or cancellation requests. "
            "Link User -> reported_issue -> Issue, Issue -> affects_order -> Order, User -> requested_action -> Activity."
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
        "extraction": ("Admin mode does not extract triplets."),
    },
}


class ExtractedEntity(BaseModel):
    id: str | None = None
    name: str
    entity_type: str = "Concept"


class ExtractedTriplet(BaseModel):
    subject_id: str | None = None
    relation: str
    object_id: str | None = None
    confidence: float | None = None
    subject: str | None = None
    subject_type: str | None = None
    object: str | None = None
    object_type: str | None = None


class ExtractionOutput(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list)
    triplets: list[ExtractedTriplet] = Field(default_factory=list)


@dataclass
class AdminAgentDeps:
    graph_tools: AdminGraphTools
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    max_tool_rounds: int


class LLMExtractor:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        default_confidence: float,
        provider: str | None = None,
    ) -> None:
        self.default_confidence = default_confidence
        self.model = self._build_model(api_key=api_key, base_url=base_url, model=model, provider=provider)

        self.assistant_agent = Agent(self.model)
        self.extraction_agent = Agent(self.model, output_type=ExtractionOutput)
        self.admin_agent: Agent[AdminAgentDeps, str] = Agent(self.model, deps_type=AdminAgentDeps)
        self._register_admin_tools()

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
        prompt = self._conversation_prompt(message=message, history=history)
        result = self.assistant_agent.run_sync(
            prompt,
            instructions=[self._assistant_system_prompt(user_name=user_name, prompt_profile=prompt_profile)],
            model_settings={"temperature": 0.4},
        )
        reply = str(result.output).strip()
        return reply or "Posso ajudar a detalhar isso melhor se quiser."

    def stream_assistant_reply(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        prompt = self._conversation_prompt(message=message, history=history)
        # NOTE: run_stream_sync in this pydantic-ai version does not accept `instructions`.
        # Build a scoped agent with a dynamic system prompt for this request.
        scoped_stream_agent = Agent(self.model, system_prompt=self._assistant_system_prompt(user_name, prompt_profile))
        stream = scoped_stream_agent.run_stream_sync(
            prompt,
            model_settings={"temperature": 0.4},
        )
        for token in stream.stream_text(delta=True):
            if token:
                yield token

    def extract_triplets(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        history: list[dict] | None = None,
    ) -> list[Triplet]:
        extraction_request = (
            f"The customer's real name is {user_name}. When the speaker refers to themselves, use the same "
            f"single entity named {user_name} with entity_type User. "
            "Extract ALL entities and relations visible in the full conversation below. "
            "Return a JSON object with keys 'entities' and 'triplets' only.\n\n"
            f"{self._conversation_prompt(message=message, history=history)}"
        )

        try:
            result = self.extraction_agent.run_sync(
                extraction_request,
                instructions=[EXTRACTION_PROMPT, self._extraction_system_prompt(prompt_profile)],
                model_settings={"temperature": 0},
            )
        except Exception:
            logger.exception("triplet_extraction_failed")
            return []

        parsed = result.output.model_dump(mode="python")
        triplets = self._from_entity_graph(parsed)
        if not triplets:
            return []
        return self._apply_speaker_name(self._reconcile_entity_types(triplets), user_name)

    def run_admin_assistant_with_tools(
        self,
        message: str,
        user_name: str,
        history: list[dict] | None,
        graph_tools: AdminGraphTools,
        max_tool_rounds: int = 6,
    ) -> dict[str, Any]:
        deps = AdminAgentDeps(
            graph_tools=graph_tools,
            tool_calls=[],
            tool_results=[],
            max_tool_rounds=max_tool_rounds,
        )
        prompt = self._conversation_prompt(message=message, history=history)

        instructions = [
            self._assistant_system_prompt(user_name=user_name, prompt_profile="graph_admin_assistant"),
            (
                "Use tools to answer graph questions. "
                "Playbook: call describe_graph_schema before free-form Cypher; "
                "prefer canonical graph model discovered from schema; "
                "use find_entity/neighbors for targeted exploration; "
                "use run_graph_query only after aligning labels/properties with discovered schema; "
                "if query fails with unknown label/property/relationship, refresh schema and retry with corrected Cypher; "
                "never invent labels like Quarto/Problema/TEM_PROBLEMA unless schema confirms them. "
                f"Never call tools more than {max_tool_rounds} times."
            ),
        ]

        try:
            result = self.admin_agent.run_sync(
                prompt,
                deps=deps,
                instructions=instructions,
                model_settings={"temperature": 0.1},
            )
            assistant_message = str(result.output).strip() or "Nao encontrei dados no grafo para responder com seguranca."
            return {
                "assistant_message": assistant_message,
                "tool_calls": deps.tool_calls,
                "tool_results": deps.tool_results,
            }
        except Exception as exc:
            logger.exception("admin_assistant_failed")
            return {
                "assistant_message": (
                    "Nao consegui usar as tools de grafo com o provedor atual. "
                    "Verifique suporte a function-calling/ferramentas no modelo configurado."
                ),
                "tool_calls": deps.tool_calls,
                "tool_results": deps.tool_results
                + [
                    {
                        "tool_name": "tool_runtime",
                        "ok": False,
                        "result": {"error": str(exc)},
                        "duration_ms": 0,
                    }
                ],
            }

    def _register_admin_tools(self) -> None:
        @self.admin_agent.tool
        def describe_graph_schema(ctx: RunContext[AdminAgentDeps]) -> dict[str, Any]:
            """Inspect current graph vocabulary to ground follow-up queries."""
            args: dict[str, Any] = {}
            return self._execute_admin_tool_safely(
                ctx,
                "describe_graph_schema",
                args,
                lambda: ctx.deps.graph_tools.describe_graph_schema(),
            )

        @self.admin_agent.tool
        def find_entity(ctx: RunContext[AdminAgentDeps], name: str, entity_type: str | None = None) -> dict[str, Any]:
            """Find entities by partial name and optional entity type."""
            args = {"name": name, "entity_type": entity_type}
            return self._execute_admin_tool_safely(
                ctx,
                "find_entity",
                args,
                lambda: ctx.deps.graph_tools.find_entity(name=name, entity_type=entity_type),
            )

        @self.admin_agent.tool
        def neighbors(
            ctx: RunContext[AdminAgentDeps], entity_name: str, depth: int = 1, limit: int = 50
        ) -> dict[str, Any]:
            """Get one-hop or two-hop neighbors for an entity name."""
            args = {"entity_name": entity_name, "depth": depth, "limit": limit}
            return self._execute_admin_tool_safely(
                ctx,
                "neighbors",
                args,
                lambda: ctx.deps.graph_tools.neighbors(entity_name=entity_name, depth=depth, limit=limit),
            )

        @self.admin_agent.tool
        def recent_relations(ctx: RunContext[AdminAgentDeps], limit: int = 50) -> dict[str, Any]:
            """Return most recent relations added or updated in graph."""
            args = {"limit": limit}
            return self._execute_admin_tool_safely(
                ctx,
                "recent_relations",
                args,
                lambda: ctx.deps.graph_tools.recent_relations(limit=limit),
            )

        @self.admin_agent.tool
        def run_graph_query(
            ctx: RunContext[AdminAgentDeps],
            cypher: str,
            params: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Run a read-only Cypher query with validation and row limits."""
            args = {"cypher": cypher, "params": params or {}}
            return self._execute_admin_tool_safely(
                ctx,
                "run_graph_query",
                args,
                lambda: ctx.deps.graph_tools.run_graph_query(cypher=cypher, params=params or {}),
            )

    def _execute_admin_tool_safely(
        self,
        ctx: RunContext[AdminAgentDeps],
        tool_name: str,
        arguments: dict[str, Any],
        operation: Callable[[], ToolExecution],
    ) -> dict[str, Any]:
        if len(ctx.deps.tool_calls) >= ctx.deps.max_tool_rounds:
            exhausted = ToolExecution(
                tool_name=tool_name,
                ok=False,
                result={
                    "error": (
                        "Tool budget exhausted. Use previously returned tool results to finish the answer "
                        "without calling more tools."
                    )
                },
                duration_ms=0,
            )
            return self._record_admin_tool_call(ctx, tool_name, arguments, exhausted)

        try:
            execution = operation()
        except Exception as exc:
            logger.exception("admin_tool_execution_failed tool=%s", tool_name)
            execution = ToolExecution(
                tool_name=tool_name,
                ok=False,
                result={"error": str(exc)},
                duration_ms=0,
            )

        return self._record_admin_tool_call(ctx, tool_name, arguments, execution)

    def _record_admin_tool_call(
        self,
        ctx: RunContext[AdminAgentDeps],
        tool_name: str,
        arguments: dict[str, Any],
        execution: ToolExecution,
    ) -> dict[str, Any]:
        ctx.deps.tool_calls.append({"tool_name": tool_name, "arguments": arguments})

        payload = {
            "tool_name": execution.tool_name,
            "ok": execution.ok,
            "result": execution.result,
            "duration_ms": execution.duration_ms,
        }
        ctx.deps.tool_results.append(payload)
        return payload

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

            relation = str(item.get("relation", "")).strip()
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

    def _conversation_prompt(self, message: str, history: list[dict] | None = None) -> str:
        history_lines = [f"{h['role'].upper()}: {h['content']}" for h in (history or [])[-20:]]
        history_lines.append(f"USER: {message}")
        return "Conversation:\n" + "\n".join(history_lines)

    def _build_model(self, api_key: str, base_url: str, model: str, provider: str | None) -> Any:
        model_name, provider_name = self._normalize_model_and_provider(model=model, provider=provider)
        provider_base_url = (base_url or "").strip() or None

        if provider_name in {"google-gla", "google-vertex"}:
            google_provider = GoogleProvider(
                api_key=api_key or None,
                vertexai=(provider_name == "google-vertex"),
                base_url=provider_base_url,
            )
            return GoogleModel(model_name=model_name, provider=google_provider)

        openai_provider = OpenAIProvider(api_key=api_key or None, base_url=provider_base_url)
        return OpenAIModel(model_name=model_name, provider=openai_provider)

    def _normalize_model_and_provider(self, model: str, provider: str | None) -> tuple[str, str]:
        raw_model = (model or "").strip()
        raw_provider = (provider or "").strip().lower()

        if ":" in raw_model:
            inferred_provider, parsed_model = raw_model.split(":", 1)
            return parsed_model.strip(), self._normalize_provider_name(inferred_provider.strip())

        if "/" in raw_model:
            inferred_provider, parsed_model = raw_model.split("/", 1)
            return parsed_model.strip(), self._normalize_provider_name(inferred_provider.strip())

        if not raw_model:
            return "gpt-4o-mini", "openai"

        if raw_provider:
            return raw_model, self._normalize_provider_name(raw_provider)

        if raw_model.startswith("gemini"):
            return raw_model, "google-gla"

        return raw_model, "openai"

    @staticmethod
    def _normalize_provider_name(provider: str) -> str:
        normalized = provider.strip().lower()
        if normalized in {"gemini", "google", "google-gla"}:
            return "google-gla"
        if normalized in {"vertexai", "google-vertex"}:
            return "google-vertex"
        if normalized in {"openai", "openai-chat"}:
            return "openai"
        return normalized or "openai"
