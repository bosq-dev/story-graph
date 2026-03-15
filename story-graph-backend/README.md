# Story Graph Backend

Backend em FastAPI que recebe mensagens de chat, usa um LLM compatível com OpenAI para extrair triplas de conhecimento e grava o resultado em Neo4j com deduplicacao.

## Fluxo

1. `POST /chat/message` recebe a mensagem.
2. Mensagem do usuario e resposta do assistente sao salvas em SQLite.
3. O backend chama o LLM para extracao de triplas no formato:
	- `subject, relation, object, object_type, confidence`
4. O backend usa `MERGE` no Neo4j para reutilizar entidades e relacoes.
5. Endpoints de consulta retornam entidades e relacoes para a UI.

## Endpoints

- `GET /health`
- `POST /chat/message`
- `GET /chat/history?session_id=...`
- `GET /graph/entities`
- `GET /graph/relations`
- `GET /graph/recent`

## Entidades Permitidas

- `User`
- `Company`
- `Product`
- `Technology`
- `Feature`
- `Issue`
- `Activity`
- `Location`
- `Concept`

## Rodando sem Docker (opcional)

```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```
