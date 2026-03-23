# Story Graph Demo

Demo local que transforma conversas em linguagem natural em um grafo de conhecimento consultavel.

## O que a demo mostra

- Usuario conversa em uma UI de chat (Streamlit).
- Backend FastAPI envia cada mensagem para agentes pydantic-ai (agnosticos a provedor).
- O LLM retorna triplas estruturadas.
- O backend faz upsert dessas triplas no Neo4j com deduplicacao.
- Historico do chat fica salvo em SQLite.

## Stack

- Neo4j (grafo)
- FastAPI (backend Python)
- Streamlit (frontend)
- pydantic-ai (OpenAI, Gemini e outros provedores)
- `uv` para gerenciamento Python

## Arquitetura e Implementacao

### Visao geral do fluxo

1. Frontend Streamlit envia mensagem para `POST /chat/message` ou `POST /chat/message/stream`.
2. Backend cria/garante sessao no SQLite (`sessions` e `messages`) e recupera historico recente.
3. `LLMExtractor` executa pipeline de extracao e normalizacao de triplas.
4. `GraphRepository` faz upsert no Neo4j com deduplicacao por entidade/relacao.
5. Backend responde com mensagem do assistente + triplas extraidas + quantidade persistida.

No modo admin (`POST /admin/chat/message/stream`), o assistente usa tools de grafo e transmite eventos SSE com chamadas e resultados de tools.

### Pipeline de extracao (implementacao atual)

A pipeline principal fica em `LLMExtractor.extract_and_stage` e segue estes estagios:

1. **Extraction**: extrai triplas estruturadas com `extraction_agent` usando historico recente e prompt de dominio (`prompt_profile`).
2. **Domain policy enforcement**: aplica regras deterministicas por dominio para garantir relacoes essenciais quando o contexto e nao ambiguo.
3. **Local canonicalization**: normaliza surface forms (espacos, `_`, formato de relacao).
4. **Entity resolution**:
	 - caminho completo com `resolution_agent` + tools quando faltam matches claros no grafo;
	 - atalho local com fuzzy match quando ha match suficiente.
5. **Semantic policy gate**: `policy_agent` valida ontologia, remove ruido e reforca relacoes canonicamente.
6. **Dedupe**: remove duplicatas por chave semantica e preserva maior confianca.
7. **Upsert no grafo**: `GraphRepository.upsert_triplets` grava entidades/relacoes com metadados de origem.

### Agentes pydantic-ai

O backend instancia agentes especializados sobre o mesmo provider/model configurado:

- `assistant_agent`: resposta para o usuario final (tom empatico e objetivo).
- `extraction_agent` (output estruturado): extracao de triplas.
- `resolution_agent` (com deps/tools): resolve entidades para reuso de entidades canonicas existentes.
- `policy_agent` (output estruturado): gate semantico e de qualidade de ontologia.
- `admin_agent` (com deps/tools): analise de grafo orientada por tools.

### Perfis de prompt (dominio)

- `hotel_customer_service` (padrao)
- `online_sales_intelligence`
- `graph_admin_assistant`

Cada perfil altera instrucoes de assistente/extracao e tambem politicas de relacoes obrigatorias.

### Tools do modo admin e resolution

Tools registradas no `admin_agent`:

- `describe_graph_schema`
- `find_entity`
- `neighbors`
- `recent_relations`
- `graph_stats`
- `shortest_path`
- `run_graph_query` (somente leitura, com validacoes)

Tools registradas no `resolution_agent`:

- `describe_graph_schema`
- `find_entity`
- `neighbors`

Detalhes importantes:

- Existe limite de rodadas de tools por execucao (tool budget) para evitar loops.
- `run_graph_query` aplica validacao de seguranca para bloquear comandos de escrita/admin.
- `describe_graph_schema` usa cache curto para reduzir custo de introspeccao repetida.

### Persistencia e modelo de dados

- **SQLite (`ChatRepository`)**: historico de sessao e mensagens (`sessions`, `messages`).
- **Neo4j (`GraphRepository`)**:
	- constraint unica por `(entity_type, normalized_name)` em `:Entity`;
	- relacoes `:RELATED` com `relation_type`, `mentions_count`, timestamps e rastreio da mensagem de origem.

Tambem ha normalizacao de aliases de relacao (ex.: variantes de localizacao convergem para `is_at`).

### APIs de consulta e streaming

- `GET /graph/entities`, `GET /graph/relations`, `GET /graph/recent` para leitura do estado do grafo.
- Chat streaming (`/chat/message/stream`) emite SSE: `token`, `done`, `error`.
- Admin streaming (`/admin/chat/message/stream`) emite SSE: `tool_call`, `tool_result`, `token`, `done`, `error`.

### Providers LLM suportados

Configuracao via `LLM_PROVIDER`:

- `openai`
- `google-gla`
- `google-vertex`

O backend normaliza aliases legados (ex.: `gemini`, `google`, `vertexai`) para esses valores canonicos.

## Subir tudo com um comando

1. Ajuste `.env` a partir de `.env.example` (principalmente `LLM_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`).
2. Execute:

```bash
docker compose up --build
```

## URLs

- Frontend: `http://localhost:8501`
- Backend: `http://localhost:8000`
- Neo4j Browser: `http://localhost:7474`

## Variaveis principais

- `LLM_API_KEY`
- `LLM_BASE_URL`
- `LLM_PROVIDER`
- `LLM_MODEL`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`

Para Gemini com provider nativo via pydantic-ai, use:

- `LLM_PROVIDER=google-gla`
- `LLM_MODEL=gemini-3-flash-preview`
- `LLM_API_KEY=<sua-chave-gemini>`

Para demo com OpenAI, use:

- `LLM_PROVIDER=openai`
- `LLM_MODEL=gpt-4o-mini`
- `LLM_API_KEY=<sua-chave-openai>`

## Exemplo de mensagem e tripla

Mensagem:

- `Nossa empresa usa Kubernetes para rodar nossos servicos.`

Tripla esperada:

- `nossa empresa, uses, Kubernetes, Technology`

## Mensagens Boas Para Demo (Hotel Customer Service)

- `Oi, estou no quarto 210 e o ar-condicionado nao esta gelando.`
- `Pedi toalhas ha 40 minutos e ainda nao chegaram.`
- `Quero trocar de quarto porque tem cheiro de mofo.`
- `O chuveiro esta sem agua quente desde ontem.`
- `Fiz check-in agora e meu quarto ainda nao estava pronto.`
- `Preciso de nota fiscal da minha estadia para a empresa.`
- `O barulho no corredor do 5 andar nao me deixou dormir.`
- `Gostaria de cancelar a reserva de amanha e saber sobre reembolso.`

Essas mensagens normalmente geram entidades como `User`, `Location`, `Issue` e `Activity`, com relacoes uteis para operacao e qualidade.

## Perguntas Interessantes Para Admin

- Quais tipos de reclamacao mais aparecem?
- Quais quartos/areas acumulam mais problemas?
- Quais problemas estao ligados a pedidos de troca de quarto?
- Quais clientes pediram reembolso?
- Existe pico de reclamacoes por periodo?

## Cypher de Exemplo (Neo4j Browser)

Top issues:

```cypher
MATCH (:Entity {entity_type:'User'})-[r:RELATED {relation_type:'reported_issue'}]->(i:Entity {entity_type:'Issue'})
RETURN i.name AS issue, sum(coalesce(r.mentions_count,1)) AS total
ORDER BY total DESC
LIMIT 10;
```

Locais com mais ocorrencias:

```cypher
MATCH (u:Entity {entity_type:'User'})-[r1:RELATED {relation_type:'reported_issue'}]->(i:Entity {entity_type:'Issue'})
MATCH (i)-[r2:RELATED {relation_type:'affects_location'}]->(l:Entity {entity_type:'Location'})
RETURN l.name AS location, count(*) AS issue_events
ORDER BY issue_events DESC
LIMIT 10;
```

Pedidos de acao por tipo:

```cypher
MATCH (:Entity {entity_type:'User'})-[r:RELATED {relation_type:'requested_action'}]->(a:Entity {entity_type:'Activity'})
RETURN a.name AS action, sum(coalesce(r.mentions_count,1)) AS total
ORDER BY total DESC;
```


## Reset volumes
docker compose down -v --remove-orphans


docker compose up -d --build


docker compose logs -f --tail=0 backend