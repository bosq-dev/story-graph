from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.admin_chat import router as admin_chat_router
from app.routes.chat import router as chat_router
from app.routes.graph import router as graph_router
from app.services.chat_repository import ChatRepository
from app.services.graph_repository import GraphRepository
from app.services.llm_extractor import LLMExtractor
from app.state import AppServices


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    chat_repo = ChatRepository(settings.sqlite_path)
    graph_repo = GraphRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    llm_extractor = LLMExtractor(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        default_confidence=settings.extraction_confidence_default,
        provider=settings.llm_provider,
    )

    app.state.services = AppServices(
        chat_repo=chat_repo,
        graph_repo=graph_repo,
        llm_extractor=llm_extractor,
    )
    yield

    chat_repo.close()
    graph_repo.close()


app = FastAPI(title="Story Graph Backend", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(graph_router)
app.include_router(admin_chat_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
