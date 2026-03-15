from dataclasses import dataclass

from app.services.chat_repository import ChatRepository
from app.services.graph_repository import GraphRepository
from app.services.llm_extractor import LLMExtractor


@dataclass
class AppServices:
    chat_repo: ChatRepository
    graph_repo: GraphRepository
    llm_extractor: LLMExtractor
