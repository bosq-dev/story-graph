from loguru import logger
from dataclasses import dataclass
import json
from difflib import SequenceMatcher
from collections.abc import Callable, Mapping
from typing import TypeAlias, TypedDict, Iterator

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models import Model
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import ALLOWED_ENTITY_TYPES, get_settings
from app.schemas import Triplet
from app.services.admin_graph_tools import AdminGraphTools, ToolExecution
from app.services.graph_repository import GraphRepository


ToolArguments: TypeAlias = dict[str, object]
ToolResultData: TypeAlias = dict[str, object] | list[dict[str, object]]


class ConversationTurn(TypedDict):
    role: str
    content: str


class ToolCallRecord(TypedDict):
    tool_name: str
    arguments: ToolArguments


class ToolResultRecord(TypedDict):
    tool_name: str
    ok: bool
    result: ToolResultData
    duration_ms: int


class AdminAssistantRunResult(TypedDict):
    assistant_message: str
    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResultRecord]


class NormalizedTripletPayload(TypedDict):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float


@dataclass(frozen=True)
class DomainPolicy:
    required_relations: tuple[tuple[str, str, str], ...] = ()




EXTRACTION_PROMPT = """
You extract knowledge graph triplets from customer service conversations.
Return JSON only with key "triplets".
Required shape:
{
    "triplets": [
        {
            "subject": "...",
            "subject_type": "User|Company|Product|Technology|Feature|Issue|Activity|Location|Concept",
            "relation": "...",
            "object": "...",
            "object_type": "User|Company|Product|Technology|Feature|Issue|Activity|Location|Concept",
            "confidence": 0.0_to_1.0
        }
    ]
}

CRITICAL - graph-first extraction:
- Your primary responsibility in this system is to populate the graph. If there is any complaint, request,
    or concrete reference (order, room, product, issue), you MUST emit triplets now.
- Extract triplets immediately from PARTIAL information. Do NOT wait for complete data.
    Example: "Pedido 1234" alone -> {"triplets":[{"subject":"Customer","subject_type":"User","relation":"mentions_product","object":"Pedido 1234","object_type":"Product","confidence":0.9}]}
    Example: "tamanho errado" alone -> {"triplets":[{"subject":"Customer","subject_type":"User","relation":"reported_issue","object":"tamanho errado","object_type":"Issue","confidence":0.9}]}
- An order/ticket reference ("pedido 1234", "order #5", "ticket 99") is ALWAYS sufficient to create a Product entity.
- A complaint or problem is ALWAYS sufficient to create an Issue entity.
- An action request (troca, devolucao, reembolso, cancelamento, limpeza) is ALWAYS an Activity entity.
- Entity ontology boundary (strict):
    Entities represent real-world referents or concrete observable states.
    Do NOT model process artifacts (cases, records, tickets, protocols, internal handling steps) as entities
    unless the conversation itself is explicitly about that artifact as a business object.
    For complaints, the Issue must be the concrete symptom/problem (e.g. "mal cheiro", "barulho", "quarto sujo"),
    while process language should become relation semantics, not entity names.
- You receive the FULL recent conversation - extract ALL relevant triplets visible across all turns,
  not just the last message. Combine information from different turns into one graph.
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
- If truly no extractable information exists, return {"triplets": []}
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

RESOLUTION_PROMPT = """
You resolve canonical entities against an existing graph.

Rules:
- Use tools before deciding if entities should be reused.
- Prefer reusing an existing entity when normalized_name and entity_type match.
- For Issue entities, also resolve semantic paraphrases (e.g. "cheiro ruim" ~ "mal cheiro")
    to one canonical existing entity when they represent the same observable problem.
- To discover paraphrases, search candidates with meaningful keyword tokens (not only full phrase)
    and validate using local context such as related Location/User neighborhood when available.
- If multiple candidates exist, pick the most concrete and stable graph name already used.
- If match is truly weak/ambiguous, keep incoming canonical entity unchanged.
- Never invent entities that are not present in input or tool results.
- Preserve ontology quality during reuse: do not reuse process-artifact entities when a concrete domain entity exists.
- Return JSON with key "triplets" only.
""".strip()

SEMANTIC_POLICY_GATE_PROMPT = """
You validate and enforce graph policy and ontology quality on normalized triplets.

Rules:
- Ensure relation names are snake_case.
- Enforce types in ALLOWED_ENTITY_TYPES.
- If an Issue and Location are connected, prefer relation affects_location.
- Enforce ontology semantics globally:
    entities must be concrete domain referents; process/administrative artifacts should be normalized away.
    Convert process language into relations when possible; otherwise drop low-quality triplets.
- Keep only entities that represent a stable referent in the graph (person/company/location/product/issue/activity/etc.).
- A valid entity denotes a concrete thing/place/actor/product/technology/feature/activity or an observable issue state.
- Prefer rewriting invalid entities to the concrete referent already present in context.
- Keep relation semantics concise and avoid adding new facts.
- If uncertain, keep high-confidence concrete triplets and drop ambiguous ones.

Return JSON with keys "triplets" and "dropped_reasons".
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


DOMAIN_POLICIES: dict[str, DomainPolicy] = {
    "hotel_customer_service": DomainPolicy(
        required_relations=(
            ("User", "reported_issue", "Issue"),
            ("Issue", "affects_location", "Location"),
            ("User", "requested_action", "Activity")
        )
    ),
    "ecommerce_support": DomainPolicy(
        required_relations=(
            ("User", "reported_issue", "Issue"),
            ("User", "requested_action", "Activity"),
            ("Issue", "affects_order", "Product"),
        )
    ),
    "saas_support": DomainPolicy(),
    "graph_admin_assistant": DomainPolicy(),
}


# Keep relation semantics stable even when models output close paraphrases.
RELATION_ALIASES: dict[str, str] = {
    "is_in": "is_at",
    "located_in": "is_at",
    "located_at": "is_at",
    "is_inside": "is_at",
    "inside": "is_at",
    "in": "is_at",
    "occupies": "is_at"
}


class ExtractedTriplet(BaseModel):
    subject: str
    subject_type: str = "Concept"
    relation: str
    object: str
    object_type: str = "Concept"
    confidence: float | None = None


class ExtractionOutput(BaseModel):
    triplets: list[ExtractedTriplet] = Field(default_factory=list)


class CanonicalizedTriplet(BaseModel):
    subject: str
    subject_type: str
    relation: str
    object: str
    object_type: str
    confidence: float | None = None


class ResolutionOutput(BaseModel):
    triplets: list[CanonicalizedTriplet] = Field(default_factory=list)


class PolicyValidationOutput(BaseModel):
    triplets: list[CanonicalizedTriplet] = Field(default_factory=list)
    dropped_reasons: list[str] = Field(default_factory=list)


@dataclass
class AdminAgentDeps:
    graph_tools: AdminGraphTools
    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResultRecord]
    max_tool_rounds: int


@dataclass
class ResolutionAgentDeps:
    graph_tools: AdminGraphTools
    tool_calls: list[ToolCallRecord]
    tool_results: list[ToolResultRecord]
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
        settings = get_settings()
        self.default_confidence = default_confidence
        self.policy_min_triplet_confidence = settings.policy_min_triplet_confidence
        self.resolution_tool_max_rounds = settings.resolution_tool_max_rounds
        self.resolution_match_confidence_threshold = settings.resolution_match_confidence_threshold
        self.model = self._build_model(api_key=api_key, base_url=base_url, model=model, provider=provider)

        self.assistant_agent = Agent(self.model)
        self.extraction_agent = Agent(self.model, output_type=ExtractionOutput)
        self.resolution_agent: Agent[ResolutionAgentDeps, ResolutionOutput] = Agent(
            self.model,
            output_type=ResolutionOutput,
            deps_type=ResolutionAgentDeps,
        )
        self.policy_agent = Agent(self.model, output_type=PolicyValidationOutput)
        self.admin_agent: Agent[AdminAgentDeps, str] = Agent(self.model, deps_type=AdminAgentDeps)
        self._register_resolution_tools()
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
        history: list[ConversationTurn] | None = None,
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
        history: list[ConversationTurn] | None = None,
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
        history: list[ConversationTurn] | None = None,
    ) -> list[Triplet]:
        extraction_request = (
            f"The customer's real name is {user_name}. When the speaker refers to themselves, use the same "
            f"single entity named {user_name} with entity_type User. "
            "Extract ALL triplets visible in the full conversation below. "
            "Return a JSON object with key 'triplets' only.\n\n"
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

        triplets = [self._to_triplet(item.model_dump(mode="python")) for item in result.output.triplets]
        triplets = [triplet for triplet in triplets if triplet is not None]
        if not triplets:
            return []
        return self._apply_speaker_name(self._reconcile_entity_types(triplets), user_name)

    def extract_and_stage(
        self,
        message: str,
        user_name: str,
        prompt_profile: str,
        graph_repo: GraphRepository,
        history: list[ConversationTurn] | None = None,
    ) -> list[Triplet]:
        raw_triplets = self.extract_triplets(
            message=message,
            user_name=user_name,
            prompt_profile=prompt_profile,
            history=history,
        )
        if not raw_triplets:
            return []
        logger.info("pipeline_stage extraction triplets={}", len(raw_triplets))

        policy_enforced_triplets = self._apply_domain_policy(raw_triplets, prompt_profile)
        if len(policy_enforced_triplets) != len(raw_triplets):
            logger.info(
                "pipeline_stage domain_policy_enforcement before={} after={}",
                len(raw_triplets),
                len(policy_enforced_triplets),
            )
        raw_triplets = policy_enforced_triplets

        canonical_triplets = self._lightweight_canonicalize_triplets(raw_triplets)
        logger.info("pipeline_stage local_canonicalization triplets={}", len(canonical_triplets))

        if self._should_call_resolution_agent(canonical_triplets, graph_repo):
            resolved_triplets = self.resolve_entity_references(canonical_triplets, graph_repo, prompt_profile)
            logger.info("pipeline_stage resolution_agent triplets={}", len(resolved_triplets))
        else:
            graph_tools = AdminGraphTools(graph_repo=graph_repo, max_rows=100)
            resolved_triplets = self._reuse_existing_entities(canonical_triplets, graph_tools)
            logger.info("pipeline_stage resolution_shortcut triplets={}", len(resolved_triplets))

        validated_triplets = self.validate_policy(resolved_triplets, prompt_profile)
        logger.info("pipeline_stage semantic_policy_gate triplets={}", len(validated_triplets))
        deduped = self._dedupe_triplets(validated_triplets)
        logger.info("pipeline_stage dedupe triplets={}", len(deduped))
        return deduped

    def resolve_entity_references(
        self,
        triplets: list[Triplet],
        graph_repo: GraphRepository,
        prompt_profile: str,
    ) -> list[Triplet]:
        if not triplets:
            return []

        graph_tools = AdminGraphTools(graph_repo=graph_repo, max_rows=100)
        deps = ResolutionAgentDeps(
            graph_tools=graph_tools,
            tool_calls=[],
            tool_results=[],
            max_tool_rounds=self.resolution_tool_max_rounds,
        )
        prompt = (
            "Resolve these canonical triplets against existing entities in graph using tools.\n\n"
            + json.dumps({"triplets": [triplet.model_dump(mode="python") for triplet in triplets]}, ensure_ascii=False)
        )

        try:
            result = self.resolution_agent.run_sync(
                prompt,
                deps=deps,
                instructions=[
                    RESOLUTION_PROMPT,
                    self._extraction_system_prompt(prompt_profile),
                    (
                        "Resolution playbook: for Issue entities without exact name match, split phrase into salient tokens "
                        "and query find_entity per token (for example 'cheiro'). Compare candidate neighbors for shared "
                        "Location or repeated user reports. If semantic equivalence is high, reuse the existing issue name "
                        "already in graph instead of creating a new paraphrase node."
                    ),
                ],
                model_settings={"temperature": 0},
            )
            model_resolved = [self._to_triplet(item.model_dump(mode="python")) for item in result.output.triplets]
            resolved = [triplet for triplet in model_resolved if triplet is not None]
        except Exception:
            logger.exception("resolution_stage_failed")
            resolved = triplets

        # Final deterministic pass to aggressively reuse existing entities by normalized name.
        return self._reuse_existing_entities(resolved, graph_tools)

    def validate_policy(self, triplets: list[Triplet], prompt_profile: str) -> list[Triplet]:
        if not triplets:
            return []

        prompt = (
            "Validate these triplets against graph policy and return corrected triplets.\n\n"
            + json.dumps({"triplets": [triplet.model_dump(mode="python") for triplet in triplets]}, ensure_ascii=False)
        )
        staged: list[Triplet]
        dropped_reasons: list[str] = []
        try:
            result = self.policy_agent.run_sync(
                prompt,
                instructions=[
                    SEMANTIC_POLICY_GATE_PROMPT,
                    self._extraction_system_prompt(prompt_profile),
                ],
                model_settings={"temperature": 0},
            )
            staged_candidates = [self._to_triplet(item.model_dump(mode="python")) for item in result.output.triplets]
            staged = [triplet for triplet in staged_candidates if triplet is not None]
            dropped_reasons = [reason.strip() for reason in result.output.dropped_reasons if reason.strip()]
        except Exception:
            logger.exception("policy_stage_failed")
            staged = triplets

        if dropped_reasons:
            logger.info(
                "pipeline_stage semantic_policy_drops count={} reasons={}",
                len(dropped_reasons),
                dropped_reasons,
            )

        if not staged:
            return []

        enforced: list[Triplet] = []
        for triplet in staged:
            if triplet.confidence < self.policy_min_triplet_confidence:
                continue

            relation = self._normalize_relation_name(triplet.relation)
            if triplet.subject_type == "Location" and triplet.object_type == "Issue":
                # Normalize direction for easier querying: Issue -> affects_location -> Location.
                enforced.append(
                    Triplet(
                        subject=triplet.object,
                        subject_type="Issue",
                        relation="affects_location",
                        object=triplet.subject,
                        object_type="Location",
                        confidence=triplet.confidence,
                    )
                )
                continue

            if triplet.subject_type == "Issue" and triplet.object_type == "Location":
                relation = "affects_location"

            enforced.append(
                Triplet(
                    subject=triplet.subject,
                    subject_type=triplet.subject_type,
                    relation=relation,
                    object=triplet.object,
                    object_type=triplet.object_type,
                    confidence=triplet.confidence,
                )
            )
        return enforced

    def run_admin_assistant_with_tools(
        self,
        message: str,
        user_name: str,
        history: list[ConversationTurn] | None,
        graph_tools: AdminGraphTools,
        max_tool_rounds: int = 6,
    ) -> AdminAssistantRunResult:
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

    def _register_resolution_tools(self) -> None:
        @self.resolution_agent.tool
        def describe_graph_schema(ctx: RunContext[ResolutionAgentDeps]) -> ToolResultRecord:
            args: ToolArguments = {}
            return self._execute_resolution_tool_safely(
                ctx,
                "describe_graph_schema",
                args,
                lambda: ctx.deps.graph_tools.describe_graph_schema(),
            )

        @self.resolution_agent.tool
        def find_entity(
            ctx: RunContext[ResolutionAgentDeps],
            name: str,
            entity_type: str | None = None,
        ) -> ToolResultRecord:
            args: ToolArguments = {"name": name, "entity_type": entity_type}
            return self._execute_resolution_tool_safely(
                ctx,
                "find_entity",
                args,
                lambda: ctx.deps.graph_tools.find_entity(name=name, entity_type=entity_type),
            )

        @self.resolution_agent.tool
        def neighbors(
            ctx: RunContext[ResolutionAgentDeps],
            entity_name: str,
            depth: int = 1,
            limit: int = 50,
        ) -> ToolResultRecord:
            args: ToolArguments = {"entity_name": entity_name, "depth": depth, "limit": limit}
            return self._execute_resolution_tool_safely(
                ctx,
                "neighbors",
                args,
                lambda: ctx.deps.graph_tools.neighbors(entity_name=entity_name, depth=depth, limit=limit),
            )

    def _execute_resolution_tool_safely(
        self,
        ctx: RunContext[ResolutionAgentDeps],
        tool_name: str,
        arguments: ToolArguments,
        operation: Callable[[], ToolExecution],
    ) -> ToolResultRecord:
        if len(ctx.deps.tool_calls) >= ctx.deps.max_tool_rounds:
            exhausted = ToolExecution(
                tool_name=tool_name,
                ok=False,
                result={"error": "Tool budget exhausted for resolution stage."},
                duration_ms=0,
            )
            return self._record_resolution_tool_call(ctx, tool_name, arguments, exhausted)

        try:
            execution = operation()
        except Exception as exc:
            logger.exception("resolution_tool_execution_failed tool={}", tool_name)
            execution = ToolExecution(
                tool_name=tool_name,
                ok=False,
                result={"error": str(exc)},
                duration_ms=0,
            )
        return self._record_resolution_tool_call(ctx, tool_name, arguments, execution)

    def _record_resolution_tool_call(
        self,
        ctx: RunContext[ResolutionAgentDeps],
        tool_name: str,
        arguments: ToolArguments,
        execution: ToolExecution,
    ) -> ToolResultRecord:
        ctx.deps.tool_calls.append({"tool_name": tool_name, "arguments": arguments})
        payload: ToolResultRecord = {
            "tool_name": execution.tool_name,
            "ok": execution.ok,
            "result": execution.result,
            "duration_ms": execution.duration_ms,
        }
        ctx.deps.tool_results.append(payload)
        return payload

    def _register_admin_tools(self) -> None:
        @self.admin_agent.tool
        def describe_graph_schema(ctx: RunContext[AdminAgentDeps]) -> ToolResultRecord:
            """Inspect current graph vocabulary to ground follow-up queries."""
            args: ToolArguments = {}
            return self._execute_admin_tool_safely(
                ctx,
                "describe_graph_schema",
                args,
                lambda: ctx.deps.graph_tools.describe_graph_schema(),
            )

        @self.admin_agent.tool
        def find_entity(ctx: RunContext[AdminAgentDeps], name: str, entity_type: str | None = None) -> ToolResultRecord:
            """Find entities by partial name and optional entity type."""
            args: ToolArguments = {"name": name, "entity_type": entity_type}
            return self._execute_admin_tool_safely(
                ctx,
                "find_entity",
                args,
                lambda: ctx.deps.graph_tools.find_entity(name=name, entity_type=entity_type),
            )

        @self.admin_agent.tool
        def neighbors(
            ctx: RunContext[AdminAgentDeps], entity_name: str, depth: int = 1, limit: int = 50
        ) -> ToolResultRecord:
            """Get one-hop or two-hop neighbors for an entity name."""
            args: ToolArguments = {"entity_name": entity_name, "depth": depth, "limit": limit}
            return self._execute_admin_tool_safely(
                ctx,
                "neighbors",
                args,
                lambda: ctx.deps.graph_tools.neighbors(entity_name=entity_name, depth=depth, limit=limit),
            )

        @self.admin_agent.tool
        def recent_relations(ctx: RunContext[AdminAgentDeps], limit: int = 50) -> ToolResultRecord:
            """Return most recent relations added or updated in graph."""
            args: ToolArguments = {"limit": limit}
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
            params: dict[str, object] | None = None,
        ) -> ToolResultRecord:
            """Run a read-only Cypher query with validation and row limits."""
            args: ToolArguments = {"cypher": cypher, "params": params or {}}
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
        arguments: ToolArguments,
        operation: Callable[[], ToolExecution],
    ) -> ToolResultRecord:
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
            logger.exception("admin_tool_execution_failed tool={}", tool_name)
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
        arguments: ToolArguments,
        execution: ToolExecution,
    ) -> ToolResultRecord:
        ctx.deps.tool_calls.append({"tool_name": tool_name, "arguments": arguments})

        payload: ToolResultRecord = {
            "tool_name": execution.tool_name,
            "ok": execution.ok,
            "result": execution.result,
            "duration_ms": execution.duration_ms,
        }
        ctx.deps.tool_results.append(payload)
        return payload

    def _to_triplet(self, payload: Mapping[str, object]) -> Triplet | None:
        normalized = self._normalize_item(payload)
        if normalized is None:
            return None
        try:
            return Triplet(**normalized)
        except Exception:
            return None

    def _lightweight_canonicalize_triplets(self, triplets: list[Triplet]) -> list[Triplet]:
        canonicalized: list[Triplet] = []
        for triplet in triplets:
            canonicalized.append(
                Triplet(
                    subject=self._canonicalize_entity_surface(triplet.subject),
                    subject_type=triplet.subject_type,
                    relation=self._normalize_relation_name(triplet.relation),
                    object=self._canonicalize_entity_surface(triplet.object),
                    object_type=triplet.object_type,
                    confidence=triplet.confidence,
                )
            )
        return canonicalized

    def _should_call_resolution_agent(self, triplets: list[Triplet], graph_repo: GraphRepository) -> bool:
        if not triplets:
            return False

        graph_tools = AdminGraphTools(graph_repo=graph_repo, max_rows=100)
        seen_entities: set[tuple[str, str]] = set()
        missing_match = False

        for triplet in triplets:
            seen_entities.add((triplet.subject_type, self._normalize_entity_name(triplet.subject)))
            seen_entities.add((triplet.object_type, self._normalize_entity_name(triplet.object)))

        for entity_type, normalized_name in seen_entities:
            has_match = self._has_entity_match(
                entity_name=normalized_name,
                entity_type=entity_type,
                graph_tools=graph_tools,
            )
            if not has_match:
                missing_match = True
                break

        return missing_match

    def _reuse_existing_entities(self, triplets: list[Triplet], graph_tools: AdminGraphTools) -> list[Triplet]:
        resolved: list[Triplet] = []
        cache: dict[tuple[str, str], tuple[str, str]] = {}

        for triplet in triplets:
            subject_name, subject_type = self._lookup_existing_entity(
                triplet.subject,
                triplet.subject_type,
                graph_tools,
                cache,
            )
            object_name, object_type = self._lookup_existing_entity(
                triplet.object,
                triplet.object_type,
                graph_tools,
                cache,
            )
            resolved.append(
                Triplet(
                    subject=subject_name,
                    subject_type=subject_type,
                    relation=self._normalize_relation_name(triplet.relation),
                    object=object_name,
                    object_type=object_type,
                    confidence=triplet.confidence,
                )
            )
        return resolved

    def _lookup_existing_entity(
        self,
        name: str,
        entity_type: str,
        graph_tools: AdminGraphTools,
        cache: dict[tuple[str, str], tuple[str, str]],
    ) -> tuple[str, str]:
        normalized_name = self._normalize_entity_name(name)
        cache_key = (entity_type, normalized_name)
        if cache_key in cache:
            return cache[cache_key]

        execution = graph_tools.find_entity(name=name, entity_type=entity_type)
        if execution.ok:
            best_name, best_type, best_score = self._best_entity_match(
                entity_name=name,
                entity_type=entity_type,
                rows=self._tool_rows(execution),
            )
            if best_name and best_type and best_score >= self.resolution_match_confidence_threshold:
                cache[cache_key] = (best_name, best_type)
                return cache[cache_key]

        cache[cache_key] = (name, entity_type)
        return cache[cache_key]

    def _tool_rows(self, execution: ToolExecution) -> list[dict[str, object]]:
        if not execution.ok or not isinstance(execution.result, dict):
            return []
        rows = execution.result.get("rows")
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def _has_entity_match(
        self,
        entity_name: str,
        entity_type: str,
        graph_tools: AdminGraphTools,
    ) -> bool:
        execution = graph_tools.find_entity(name=entity_name, entity_type=entity_type)
        if not execution.ok:
            return False
        _, _, score = self._best_entity_match(
            entity_name=entity_name,
            entity_type=entity_type,
            rows=self._tool_rows(execution),
        )
        return score >= self.resolution_match_confidence_threshold

    def _best_entity_match(
        self,
        entity_name: str,
        entity_type: str,
        rows: list[dict[str, object]],
    ) -> tuple[str | None, str | None, float]:
        target = self._normalize_entity_name(entity_name)
        best_name: str | None = None
        best_type: str | None = None
        best_score = 0.0

        for row in rows:
            row_type = str(row.get("entity_type", "")).strip()
            row_name = str(row.get("name", "")).strip()
            row_normalized = str(row.get("normalized_name", "")).strip() or self._normalize_entity_name(row_name)
            if not row_name or row_type != entity_type:
                continue

            score = self._entity_match_score(target=target, candidate=self._normalize_entity_name(row_normalized))
            if score > best_score:
                best_score = score
                best_name = row_name
                best_type = row_type

        return best_name, best_type, best_score

    @staticmethod
    def _entity_match_score(target: str, candidate: str) -> float:
        if not target or not candidate:
            return 0.0
        if target == candidate:
            return 1.0

        # Blend edit similarity and token overlap for stable fuzzy reuse decisions.
        seq_ratio = SequenceMatcher(None, target, candidate).ratio()
        target_tokens = set(target.split())
        candidate_tokens = set(candidate.split())
        if not target_tokens or not candidate_tokens:
            token_jaccard = 0.0
        else:
            token_jaccard = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)

        return (seq_ratio * 0.7) + (token_jaccard * 0.3)

    def _apply_domain_policy(self, triplets: list[Triplet], prompt_profile: str) -> list[Triplet]:
        if not triplets:
            return []

        profile_key = self.resolve_prompt_profile(prompt_profile)
        policy = DOMAIN_POLICIES.get(profile_key)
        if policy is None or not policy.required_relations:
            return triplets

        enriched = list(triplets)
        existing_keys = {
            (
                triplet.subject_type,
                self._normalize_entity_name(triplet.subject),
                self._normalize_relation_name(triplet.relation),
                triplet.object_type,
                self._normalize_entity_name(triplet.object),
            )
            for triplet in triplets
        }

        entities_by_type: dict[str, set[str]] = {}
        for triplet in triplets:
            entities_by_type.setdefault(triplet.subject_type, set()).add(triplet.subject)
            entities_by_type.setdefault(triplet.object_type, set()).add(triplet.object)

        for subject_type, relation, object_type in policy.required_relations:
            subjects = sorted(entities_by_type.get(subject_type, set()))
            objects = sorted(entities_by_type.get(object_type, set()))

            # Deterministic safety: only auto-link when relation is unambiguous in context.
            if len(subjects) != 1 or len(objects) != 1:
                continue

            subject = subjects[0]
            obj = objects[0]
            key = (
                subject_type,
                self._normalize_entity_name(subject),
                self._normalize_relation_name(relation),
                object_type,
                self._normalize_entity_name(obj),
            )
            if key in existing_keys:
                continue

            enriched.append(
                Triplet(
                    subject=subject,
                    subject_type=subject_type,
                    relation=relation,
                    object=obj,
                    object_type=object_type,
                    confidence=self.default_confidence,
                )
            )
            existing_keys.add(key)

        return enriched

    def _dedupe_triplets(self, triplets: list[Triplet]) -> list[Triplet]:
        deduped: dict[tuple[str, str, str, str, str], Triplet] = {}
        for triplet in triplets:
            key = (
                triplet.subject_type,
                self._normalize_entity_name(triplet.subject),
                self._normalize_relation_name(triplet.relation),
                triplet.object_type,
                self._normalize_entity_name(triplet.object),
            )
            existing = deduped.get(key)
            if existing is None or triplet.confidence > existing.confidence:
                deduped[key] = Triplet(
                    subject=triplet.subject,
                    subject_type=triplet.subject_type,
                    relation=self._normalize_relation_name(triplet.relation),
                    object=triplet.object,
                    object_type=triplet.object_type,
                    confidence=triplet.confidence,
                )
        return list(deduped.values())

    @staticmethod
    def _normalize_relation_name(value: str) -> str:
        normalized = "_".join(value.strip().lower().replace("-", " ").split())
        return RELATION_ALIASES.get(normalized, normalized)

    @staticmethod
    def _canonicalize_entity_surface(value: str) -> str:
        return " ".join(value.strip().replace("_", " ").split())

    def _normalize_item(self, item: Mapping[str, object]) -> NormalizedTripletPayload | None:
        subject = str(item.get("subject", "")).strip()
        subject_type = str(item.get("subject_type", "Concept")).strip()
        relation = str(item.get("relation", "")).strip()
        obj = str(item.get("object", "")).strip()
        object_type = str(item.get("object_type", "Concept")).strip()
        confidence_raw = item.get("confidence", self.default_confidence)

        if not subject or not relation or not obj:
            return None

        if subject_type not in ALLOWED_ENTITY_TYPES:
            subject_type = "Concept"
        if object_type not in ALLOWED_ENTITY_TYPES:
            object_type = "Concept"

        try:
            confidence = float(str(confidence_raw))
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
        return " ".join(value.strip().lower().replace("_", " ").split())

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

    def _conversation_prompt(self, message: str, history: list[ConversationTurn] | None = None) -> str:
        history_lines = [f"{h['role'].upper()}: {h['content']}" for h in (history or [])[-20:]]
        history_lines.append(f"USER: {message}")
        return "Conversation:\n" + "\n".join(history_lines)

    def _build_model(self, api_key: str, base_url: str, model: str, provider: str | None) -> Model:
        provider_name = self._normalize_provider_name(provider)
        model_name = (model or "").strip() or ("gemini-2.5-flash" if provider_name.startswith("google") else "gpt-4o-mini")
        provider_base_url = (base_url or "").strip() or None

        if provider_name in {"google-gla", "google-vertex"}:
            google_provider = GoogleProvider(
                api_key=api_key or None,
                vertexai=(provider_name == "google-vertex"),
                base_url=provider_base_url,
            )
            return GoogleModel(model_name=model_name, provider=google_provider)

        if provider_name != "openai":
            raise ValueError(
                "Unsupported LLM provider. Use one of: openai, google-gla, google-vertex "
                "(configured via LLM_PROVIDER)."
            )

        openai_provider = OpenAIProvider(api_key=api_key or None, base_url=provider_base_url)
        return OpenAIResponsesModel(model_name=model_name, provider=openai_provider)

    @staticmethod
    def _normalize_provider_name(provider: str | None) -> str:
        normalized = (provider or "").strip().lower()
        if normalized in {"gemini", "google", "google-gla"}:
            return "google-gla"
        if normalized in {"vertexai", "google-vertex"}:
            return "google-vertex"
        if normalized in {"openai", "openai-chat"}:
            return "openai"
        return normalized or "openai"
