from datetime import datetime, timezone
from typing import Any

from neo4j import GraphDatabase, Session
from loguru import logger

from app.config import ALLOWED_ENTITY_TYPES
from app.schemas import Triplet


RELATION_ALIASES: dict[str, str] = {
    "is_in": "is_at",
    "located_in": "is_at",
    "located_at": "is_at",
    "is_at_location": "is_at",
    "at_location": "is_at",
    "in_location": "is_at",
    "is_inside": "is_at",
    "inside": "is_at",
    "in": "is_at",
}


class GraphRepository:
    def __init__(self, uri: str, username: str, password: str, database: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self.database = database
        self._init_schema()

    def _init_schema(self) -> None:
        constraint_query = (
            "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE (e.entity_type, e.normalized_name) IS UNIQUE"
        )
        with self._driver.session(database=self.database) as session:
            session.run(constraint_query)
            self._run_normalization_maintenance(session)

    def _run_normalization_maintenance(self, session: Session) -> None:
        renamed_count = int(
            (
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.normalized_name CONTAINS '_'
                    WITH e, trim(replace(toLower(e.normalized_name), '_', ' ')) AS target_normalized
                    WHERE target_normalized <> e.normalized_name
                      AND NOT EXISTS {
                        MATCH (:Entity {entity_type: e.entity_type, normalized_name: target_normalized})
                      }
                    SET e.normalized_name = target_normalized
                    RETURN count(e) AS renamed_count
                    """
                ).single()
                or {"renamed_count": 0}
            )["renamed_count"]
        )

        migrated_relation_count = int(
            (
                session.run(
                    """
                    MATCH (s:Entity)-[r:RELATED]->(o:Entity)
                    WHERE r.relation_type IN $legacy_relation_types
                    MERGE (s)-[m:RELATED {relation_type: $canonical_relation_type}]->(o)
                      ON CREATE SET
                        m.created_at = coalesce(r.created_at, $timestamp),
                        m.updated_at = coalesce(r.updated_at, $timestamp),
                        m.first_source_message_id = r.first_source_message_id,
                        m.first_source_message = r.first_source_message,
                        m.last_source_message_id = r.last_source_message_id,
                        m.last_source_message = r.last_source_message,
                        m.first_confidence = r.first_confidence,
                        m.last_confidence = r.last_confidence,
                        m.mentions_count = coalesce(r.mentions_count, 1)
                      ON MATCH SET
                        m.updated_at = coalesce(r.updated_at, m.updated_at),
                        m.last_source_message_id = coalesce(r.last_source_message_id, m.last_source_message_id),
                        m.last_source_message = coalesce(r.last_source_message, m.last_source_message),
                        m.last_confidence = coalesce(r.last_confidence, m.last_confidence),
                        m.mentions_count = coalesce(m.mentions_count, 1) + coalesce(r.mentions_count, 1)
                    DELETE r
                    RETURN count(*) AS migrated_relation_count
                    """,
                    legacy_relation_types=list(RELATION_ALIASES.keys()),
                    canonical_relation_type="is_at",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ).single()
                or {"migrated_relation_count": 0}
            )["migrated_relation_count"]
        )

        if renamed_count or migrated_relation_count:
            logger.info(
                "graph_normalization_maintenance renamed_entities={} migrated_relations={}",
                renamed_count,
                migrated_relation_count,
            )

    @staticmethod
    def _normalize_entity(value: str) -> str:
        return " ".join(value.strip().lower().replace("_", " ").split())

    @staticmethod
    def _normalize_relation(value: str) -> str:
        normalized = "_".join(value.strip().lower().replace("-", " ").split())
        return RELATION_ALIASES.get(normalized, normalized)

    @staticmethod
    def _resolve_entity_type(
        session: Session,
        normalized_name: str,
        requested_type: str,
        type_cache: dict[str, str],
    ) -> str:
        if normalized_name in type_cache:
            return type_cache[normalized_name]

        result = session.run(
            """
            MATCH (e:Entity {normalized_name: $normalized_name})
            RETURN collect(DISTINCT e.entity_type) AS existing_types
            """,
            normalized_name=normalized_name,
        ).single()
        existing_types = result["existing_types"] if result and result["existing_types"] else []

        # Promote prior generic Concept entities when a stronger resolved type arrives,
        # as long as target (normalized_name, requested_type) does not already exist.
        if (
            requested_type != "Concept"
            and "Concept" in existing_types
            and requested_type not in existing_types
        ):
            promotion_result = session.run(
                """
                MATCH (e:Entity {normalized_name: $normalized_name, entity_type: 'Concept'})
                OPTIONAL MATCH (target:Entity {normalized_name: $normalized_name, entity_type: $requested_type})
                WITH e, target
                WHERE target IS NULL
                SET e.entity_type = $requested_type
                RETURN count(e) AS promoted_count
                """,
                normalized_name=normalized_name,
                requested_type=requested_type,
            ).single()
            promoted_count = int(promotion_result["promoted_count"]) if promotion_result else 0
            if promoted_count > 0:
                existing_types = [requested_type if value == "Concept" else value for value in existing_types]

        if requested_type in existing_types:
            resolved = requested_type
        elif existing_types:
            non_concept = [value for value in existing_types if value != "Concept"]
            resolved = non_concept[0] if non_concept else existing_types[0]
        else:
            resolved = requested_type

        type_cache[normalized_name] = resolved
        return resolved

    def upsert_triplets(self, triplets: list[Triplet], source_message_id: str, source_message: str) -> int:
        if not triplets:
            return 0

        now = datetime.now(timezone.utc).isoformat()
        query = """
        MERGE (s:Entity {entity_type: $subject_type, normalized_name: $subject_normalized})
          ON CREATE SET s.name = $subject, s.created_at = $timestamp
        MERGE (o:Entity {entity_type: $object_type, normalized_name: $object_normalized})
          ON CREATE SET o.name = $object, o.created_at = $timestamp
        MERGE (s)-[r:RELATED {relation_type: $relation_type}]->(o)
          ON CREATE SET
            r.created_at = $timestamp,
            r.updated_at = $timestamp,
            r.first_source_message_id = $source_message_id,
            r.first_source_message = $source_message,
            r.last_source_message_id = $source_message_id,
            r.last_source_message = $source_message,
            r.first_confidence = $confidence,
            r.last_confidence = $confidence,
            r.mentions_count = 1
          ON MATCH SET
            r.updated_at = $timestamp,
            r.last_source_message_id = $source_message_id,
            r.last_source_message = $source_message,
            r.last_confidence = $confidence,
            r.mentions_count = coalesce(r.mentions_count, 1) + 1
        """

        with self._driver.session(database=self.database) as session:
            type_cache: dict[str, str] = {}
            for triple in triplets:
                requested_subject_type = triple.subject_type if triple.subject_type in ALLOWED_ENTITY_TYPES else "Concept"
                requested_object_type = triple.object_type if triple.object_type in ALLOWED_ENTITY_TYPES else "Concept"
                subject_normalized = self._normalize_entity(triple.subject)
                object_normalized = self._normalize_entity(triple.object)

                subject_type = self._resolve_entity_type(
                    session=session,
                    normalized_name=subject_normalized,
                    requested_type=requested_subject_type,
                    type_cache=type_cache,
                )
                object_type = self._resolve_entity_type(
                    session=session,
                    normalized_name=object_normalized,
                    requested_type=requested_object_type,
                    type_cache=type_cache,
                )

                session.run(
                    query,
                    subject=triple.subject,
                    subject_type=subject_type,
                    subject_normalized=subject_normalized,
                    relation_type=self._normalize_relation(triple.relation),
                    object=triple.object,
                    object_type=object_type,
                    object_normalized=object_normalized,
                    source_message_id=source_message_id,
                    source_message=source_message,
                    confidence=float(triple.confidence),
                    timestamp=now,
                )
        return len(triplets)

    def list_entities(self, limit: int = 200) -> list[dict[str, Any]]:
        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.normalized_name AS normalized_name, e.entity_type AS entity_type
        ORDER BY e.entity_type, e.name
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def list_relations(self, limit: int = 200) -> list[dict[str, Any]]:
        query = """
        MATCH (s:Entity)-[r:RELATED]->(o:Entity)
        RETURN
          s.name AS subject,
          s.entity_type AS subject_type,
          r.relation_type AS relation,
          o.name AS object,
          o.entity_type AS object_type,
          coalesce(r.mentions_count, 1) AS mentions_count,
          r.created_at AS created_at,
          r.updated_at AS updated_at,
          r.last_confidence AS last_confidence
        ORDER BY r.updated_at DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def list_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        query = """
        MATCH (s:Entity)-[r:RELATED]->(o:Entity)
        RETURN
          s.name AS subject,
          r.relation_type AS relation,
          o.name AS object,
          r.last_source_message_id AS source_message_id,
          r.updated_at AS updated_at
        ORDER BY r.updated_at DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def list_relations_for_message_ids(self, message_ids: list[str], limit: int = 80) -> list[dict[str, Any]]:
        if not message_ids:
            return []

        safe_limit = max(1, min(int(limit), 500))
        query = """
        MATCH (s:Entity)-[r:RELATED]->(o:Entity)
        WHERE r.last_source_message_id IN $message_ids
           OR r.first_source_message_id IN $message_ids
        RETURN
          s.name AS subject,
          s.entity_type AS subject_type,
          r.relation_type AS relation,
          o.name AS object,
          o.entity_type AS object_type,
          r.last_confidence AS confidence,
          r.updated_at AS updated_at
        ORDER BY r.updated_at DESC
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, message_ids=message_ids, limit=safe_limit)
            return [record.data() for record in result]

    def close(self) -> None:
        self._driver.close()

    def run_readonly_query(
        self,
        cypher: str,
        params: dict | None = None,
        row_limit: int = 200,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(row_limit), 500))
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, **(params or {}))  # type: ignore[arg-type]
            rows: list[dict[str, Any]] = []
            for index, record in enumerate(result):
                if index >= safe_limit:
                    break
                rows.append(record.data())
            return rows
