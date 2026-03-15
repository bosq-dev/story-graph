from fastapi import APIRouter, Depends, Request

from app.schemas import EntityItem, GraphRecentItem, RelationItem
from app.state import AppServices

router = APIRouter(prefix="/graph", tags=["graph"])


def get_services(request: Request) -> AppServices:
    return request.app.state.services


@router.get("/entities", response_model=list[EntityItem])
def list_entities(limit: int = 200, services: AppServices = Depends(get_services)) -> list[EntityItem]:
    entities = services.graph_repo.list_entities(limit=limit)
    return [EntityItem(**item) for item in entities]


@router.get("/relations", response_model=list[RelationItem])
def list_relations(limit: int = 200, services: AppServices = Depends(get_services)) -> list[RelationItem]:
    relations = services.graph_repo.list_relations(limit=limit)
    return [RelationItem(**item) for item in relations]


@router.get("/recent", response_model=list[GraphRecentItem])
def list_recent(limit: int = 20, services: AppServices = Depends(get_services)) -> list[GraphRecentItem]:
    recent = services.graph_repo.list_recent(limit=limit)
    return [GraphRecentItem(**item) for item in recent]
