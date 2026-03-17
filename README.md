# Story Graph Demo

Demo local que transforma conversas em linguagem natural em um grafo de conhecimento consultavel.

## O que a demo mostra

- Usuario conversa em uma UI de chat (Streamlit).
- Backend FastAPI envia cada mensagem para um LLM via SDK OpenAI compativel.
- O LLM retorna triplas estruturadas.
- O backend faz upsert dessas triplas no Neo4j com deduplicacao.
- Historico do chat fica salvo em SQLite.

## Stack

- Neo4j (grafo)
- FastAPI (backend Python)
- Streamlit (frontend)
- OpenAI SDK (endpoint OpenAI ou compativel, como Gemini)
- `uv` para gerenciamento Python

## Subir tudo com um comando

1. Ajuste `.env` a partir de `.env.example` (principalmente `OPENAI_API_KEY`).
2. Execute:

```bash
docker compose up --build
```

## URLs

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`

## Variaveis principais

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

Para Gemini via endpoint compativel OpenAI, use:

- `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/openai/`
- `OPENAI_MODEL=gemini-2.0-flash`
- `OPENAI_API_KEY=<sua-chave-gemini>`

## Exemplo de mensagem e tripla

Mensagem:

- `Nossa empresa usa Kubernetes para rodar nossos servicos.`

Tripla esperada:

- `nossa empresa, uses, Kubernetes, Technology`


## Reset volumes
docker compose down -v --remove-orphans


docker compose up -d --build


docker compose logs -f --tail=0 backend