from datetime import datetime, timezone

from neo4j import GraphDatabase

from app.config import ALLOWED_ENTITY_TYPES
from app.schemas import Triplet


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

    @staticmethod
    def _normalize_entity(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _normalize_relation(value: str) -> str:
        return "_".join(value.strip().lower().split())

    @staticmethod
    def _resolve_entity_type(
        session,
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

    def list_entities(self, limit: int = 200) -> list[dict]:
        query = """
        MATCH (e:Entity)
        RETURN e.name AS name, e.normalized_name AS normalized_name, e.entity_type AS entity_type
        ORDER BY e.entity_type, e.name
        LIMIT $limit
        """
        with self._driver.session(database=self.database) as session:
            result = session.run(query, limit=limit)
            return [record.data() for record in result]

    def list_relations(self, limit: int = 200) -> list[dict]:
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

    def list_recent(self, limit: int = 20) -> list[dict]:
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

    def close(self) -> None:
        self._driver.close()
