import re
import time
from dataclasses import dataclass
from typing import Any

from app.services.graph_repository import GraphRepository
from loguru import logger

_FORBIDDEN_PATTERN = re.compile(
    r"\b(create|merge|delete|detach|set|remove|drop|call\s+dbms|apoc|load\s+csv|foreach|grant|deny|revoke)\b",
    flags=re.IGNORECASE,
)


@dataclass
class ToolExecution:
    tool_name: str
    ok: bool
    result: list[dict] | dict[str, Any]
    duration_ms: int


class AdminGraphTools:
    def __init__(self, graph_repo: GraphRepository, max_rows: int = 200) -> None:
        self.graph_repo = graph_repo
        self.max_rows = max_rows

    def run_graph_query(self, cypher: str, params: dict | None = None) -> ToolExecution:
        started = time.perf_counter()
        try:
            sanitized = self._sanitize_readonly_cypher(cypher)
            safe_params = self._validate_params(params or {})
            rows = self.graph_repo.run_readonly_query(
                cypher=sanitized,
                params=safe_params,
                row_limit=self.max_rows,
            )
            return ToolExecution(
                tool_name="run_graph_query",
                ok=True,
                result={"rows": rows, "row_count": len(rows)},
                duration_ms=self._duration_ms(started),
            )
        except Exception as exc:
            logger.warning("admin_tool_run_graph_query_rejected error={}", exc)
            return ToolExecution(
                tool_name="run_graph_query",
                ok=False,
                result={"error": str(exc)},
                duration_ms=self._duration_ms(started),
            )

    def find_entity(self, name: str, entity_type: str | None = None) -> ToolExecution:
        started = time.perf_counter()
        query = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($name)
          AND ($entity_type IS NULL OR e.entity_type = $entity_type)
        RETURN e.name AS name, e.entity_type AS entity_type, e.normalized_name AS normalized_name
        ORDER BY e.entity_type, e.name
        LIMIT $limit
        """
        rows = self.graph_repo.run_readonly_query(
            cypher=query,
            params={"name": name.strip(), "entity_type": entity_type, "limit": self.max_rows},
            row_limit=self.max_rows,
        )
        return ToolExecution(
            tool_name="find_entity",
            ok=True,
            result={"rows": rows, "row_count": len(rows)},
            duration_ms=self._duration_ms(started),
        )

    def neighbors(self, entity_name: str, depth: int = 1, limit: int = 50) -> ToolExecution:
        started = time.perf_counter()
        safe_depth = max(1, min(int(depth), 2))
        safe_limit = max(1, min(int(limit), self.max_rows))

        if safe_depth == 1:
            query = """
            MATCH (e:Entity {normalized_name: $normalized_name})-[r:RELATED]-(n:Entity)
                        RETURN
                            e.name AS center,
                            coalesce(r.relation_type, type(r)) AS relation_type,
                            n.name AS neighbor,
                            n.entity_type AS neighbor_type
            LIMIT $limit
            """
        else:
            query = """
            MATCH p=(e:Entity {normalized_name: $normalized_name})-[r:RELATED*1..2]-(n:Entity)
            RETURN e.name AS center, n.name AS neighbor, n.entity_type AS neighbor_type, length(p) AS hops
            LIMIT $limit
            """

        rows = self.graph_repo.run_readonly_query(
            cypher=query,
            params={"normalized_name": self._normalize_name(entity_name), "limit": safe_limit},
            row_limit=safe_limit,
        )
        return ToolExecution(
            tool_name="neighbors",
            ok=True,
            result={"rows": rows, "row_count": len(rows)},
            duration_ms=self._duration_ms(started),
        )

    def recent_relations(self, limit: int = 50) -> ToolExecution:
        started = time.perf_counter()
        safe_limit = max(1, min(int(limit), self.max_rows))
        rows = self.graph_repo.list_recent(limit=safe_limit)
        return ToolExecution(
            tool_name="recent_relations",
            ok=True,
            result={"rows": rows, "row_count": len(rows)},
            duration_ms=self._duration_ms(started),
        )

    def describe_graph_schema(self) -> ToolExecution:
        started = time.perf_counter()
        try:
            labels_rows = self.graph_repo.run_readonly_query(
                cypher="MATCH (n) WITH labels(n) AS ls UNWIND ls AS label RETURN DISTINCT label ORDER BY label LIMIT 50",
                params={},
                row_limit=50,
            )
            rel_type_rows = self.graph_repo.run_readonly_query(
                cypher="MATCH ()-[r]->() RETURN DISTINCT type(r) AS relationship_type ORDER BY relationship_type LIMIT 50",
                params={},
                row_limit=50,
            )
            entity_type_rows = self.graph_repo.run_readonly_query(
                cypher="MATCH (e:Entity) WHERE e.entity_type IS NOT NULL RETURN DISTINCT e.entity_type AS entity_type ORDER BY entity_type LIMIT 100",
                params={},
                row_limit=100,
            )
            relation_type_rows = self.graph_repo.run_readonly_query(
                cypher="MATCH (:Entity)-[r:RELATED]->(:Entity) WHERE r.relation_type IS NOT NULL RETURN DISTINCT r.relation_type AS relation_type ORDER BY relation_type LIMIT 200",
                params={},
                row_limit=200,
            )
            sample_entities = self.graph_repo.run_readonly_query(
                cypher="MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS entity_type LIMIT 20",
                params={},
                row_limit=20,
            )

            return ToolExecution(
                tool_name="describe_graph_schema",
                ok=True,
                result={
                    "labels": [row.get("label") for row in labels_rows if row.get("label")],
                    "relationship_types": [
                        row.get("relationship_type") for row in rel_type_rows if row.get("relationship_type")
                    ],
                    "entity_types": [
                        row.get("entity_type") for row in entity_type_rows if row.get("entity_type")
                    ],
                    "relation_types": [
                        row.get("relation_type") for row in relation_type_rows if row.get("relation_type")
                    ],
                    "sample_entities": sample_entities,
                },
                duration_ms=self._duration_ms(started),
            )
        except Exception as exc:
            logger.warning("admin_tool_describe_graph_schema_failed error={}", exc)
            return ToolExecution(
                tool_name="describe_graph_schema",
                ok=False,
                result={"error": str(exc)},
                duration_ms=self._duration_ms(started),
            )

    def _sanitize_readonly_cypher(self, cypher: str) -> str:
        normalized = " ".join(cypher.strip().split())
        if not normalized:
            raise ValueError("Cypher is empty.")
        if ";" in normalized:
            raise ValueError("Multiple statements are not allowed.")
        lowered = normalized.lower()
        if not (lowered.startswith("match") or lowered.startswith("with")):
            raise ValueError("Query must start with MATCH or WITH.")
        if " return " not in f" {lowered} ":
            raise ValueError("Query must include a RETURN clause.")
        if _FORBIDDEN_PATTERN.search(lowered):
            raise ValueError("Read-only policy violation: forbidden clause detected.")
        if " limit " not in f" {lowered} ":
            normalized = f"{normalized} LIMIT {self.max_rows}"
        return normalized

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise ValueError("params must be an object.")
        if len(params) > 20:
            raise ValueError("Too many query params.")

        validated: dict[str, Any] = {}
        for key, value in params.items():
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", key):
                raise ValueError(f"Invalid param name: {key}")
            validated[key] = self._validate_param_value(value)
        return validated

    def _validate_param_value(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 500:
                raise ValueError("String params cannot exceed 500 chars.")
            return value
        if isinstance(value, list):
            if len(value) > 50:
                raise ValueError("List params cannot exceed 50 items.")
            return [self._validate_param_value(item) for item in value]
        if isinstance(value, dict):
            if len(value) > 20:
                raise ValueError("Nested param objects cannot exceed 20 keys.")
            return {str(k): self._validate_param_value(v) for k, v in value.items()}
        raise ValueError("Unsupported param type.")

    @staticmethod
    def _normalize_name(value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())

    @staticmethod
    def _duration_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)
