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