# Story Graph Backend

Backend em FastAPI que recebe mensagens de chat, usa um LLM provider-agnostic (LiteLLM) para extrair triplas de conhecimento e grava o resultado em Neo4j com deduplicacao.

## Configuracao de LLM

Use as variaveis abaixo para selecionar provedor e modelo:

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_MODEL` (ex.: `openai/gpt-4o-mini`, `gemini/gemini-2.0-flash`)
- `LLM_PROVIDER` (opcional quando `LLM_MODEL` ja tem prefixo do provedor)

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
