import json
from typing import Any

from openai import OpenAI

from app.config import ALLOWED_ENTITY_TYPES
from app.schemas import Triplet


EXTRACTION_PROMPT = """
You extract knowledge graph data from user messages.
Return JSON only in this exact shape:
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
Rules:
- Reuse the same entity id when the text refers to the same real-world entity in this message.
- Do not create duplicate entities with different types for the same thing in one message.
- Example: "quarto 2" should be a single entity, usually Location.
- Keep relation short and normalized in snake_case style where possible.
- If no relevant knowledge exists, return {"entities": [], "triplets": []}
- Never add fields outside the schema.
""".strip()


class LLMExtractor:
    def __init__(self, api_key: str, base_url: str, model: str, default_confidence: float) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.default_confidence = default_confidence

    def build_assistant_reply(self, message: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=0.4,
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise assistant for product and engineering conversations.",
                },
                {"role": "user", "content": message},
            ],
        )
        content = completion.choices[0].message.content or ""
        return content.strip() or "Posso ajudar a detalhar isso melhor se quiser."

    def extract_triplets(self, message: str) -> list[Triplet]:
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract triplets from the text below. Return a JSON object in the format "
                            '{"triplets": [...]} only.\\n\\n'
                            f"Text: {message}"
                        ),
                    },
                ],
            )
        except Exception:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Extract triplets from the text below. Return only valid JSON with key 'triplets'.\\n\\n"
                            f"Text: {message}"
                        ),
                    },
                ],
            )
        raw = completion.choices[0].message.content or "{}"
        parsed = self._safe_parse_json(raw)
        if isinstance(parsed, dict):
            triplets = self._from_entity_graph(parsed)
            if triplets:
                return self._reconcile_entity_types(triplets)

            items = parsed.get("triplets", [])
            if isinstance(items, list):
                legacy = self._from_legacy_triplets(items)
                return self._reconcile_entity_types(legacy)

        if isinstance(parsed, list):
            legacy = self._from_legacy_triplets(parsed)
            return self._reconcile_entity_types(legacy)

        return []

    def _from_entity_graph(self, payload: dict[str, Any]) -> list[Triplet]:
        entities_raw = payload.get("entities", [])
        triplets_raw = payload.get("triplets", [])
        if not isinstance(entities_raw, list) or not isinstance(triplets_raw, list):
            return []

        entities: dict[str, dict[str, str]] = {}
        for item in entities_raw:
            if not isinstance(item, dict):
                continue
            entity_id = str(item.get("id", "")).strip()
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("entity_type", "Concept")).strip()
            if not entity_id or not name:
                continue
            if entity_type not in ALLOWED_ENTITY_TYPES:
                entity_type = "Concept"
            entities[entity_id] = {"name": name, "entity_type": entity_type}

        result: list[Triplet] = []
        for item in triplets_raw:
            if not isinstance(item, dict):
                continue
            subject_id = str(item.get("subject_id", "")).strip()
            object_id = str(item.get("object_id", "")).strip()
            relation = str(item.get("relation", "")).strip()
            confidence = item.get("confidence", self.default_confidence)
            if not subject_id or not object_id or not relation:
                continue
            subject_entity = entities.get(subject_id)
            object_entity = entities.get(object_id)
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

    @staticmethod
    def _safe_parse_json(raw: str) -> dict[str, Any] | list[Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"triplets": []}
