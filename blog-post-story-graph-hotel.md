# Como seus usuários usam o seu chatbot? Do texto solto ao grafo de conhecimento

## TL;DR

Insights de chatbot ainda são pouco explorados. A maior parte das soluções fala de memória em texto puro, mas quase não transforma conversa em dado estruturado.

Neste projeto, cada mensagem do chat pode virar triplas semânticas, que são persistidas em um grafo consultável. Com isso, saímos do "parece que os usuários reclamam disso" para "estes usuários, nestes locais, com estes problemas, nestes padrões".

E essa estratégia funciona para qualquer domínio em que conversas carregam sinais de negócio.

---

## De conversa para dado estruturado (independente do setor)

A ideia central é simples:

- você coleta conversas em linguagem natural,
- extrai fatos estruturados,
- conecta esses fatos em um grafo,
- e transforma histórico textual em base analítica consultável.

Esse modelo pode ser usado em vários cenários:

- atendimento ao cliente,
- suporte técnico SaaS,
- pré-vendas e discovery,
- e-commerce,
- operações internas.

Sempre que existir uma pergunta do tipo "como os usuários estão usando/sofrendo/pedindo X?", essa abordagem tende a funcionar bem.

---

## Como seus usuários usam o seu chatbot?

Essa é a pergunta central.

Hoje, times de produto e operação têm milhares de mensagens, mas pouca estrutura para responder a perguntas simples:

- quais problemas mais aparecem?
- qual local ou contexto concentra mais reclamações?
- quais pedidos estão ligados a reembolso?
- como um problema se conecta a um usuário e a uma ação solicitada?

Sem estrutura, isso vira leitura manual, amostragem parcial e decisão com baixa confiança.

Para deixar isso concreto, vamos para um exemplo real.

---

## Exemplo prático: hotel customer service

No cenário de hotel, o usuário escreve coisas como:

- "Estou no quarto 210 e o ar não está gelando"
- "Pedi toalha faz 40 minutos"
- "Quero trocar de quarto por causa de cheiro de mofo"
- "Quero cancelar e entender o reembolso"

Cada mensagem parece simples, mas o valor real está nas conexões:

- Usuário -> problema reportado
- Problema -> local afetado
- Usuário -> ação solicitada

É exatamente esse tipo de conexão que transformamos em grafo.

---

## O que é uma tripla? E o que é um grafo?

Um grafo é uma forma de representar conhecimento como uma rede:

- **nós**: as entidades (usuário, quarto, problema, atividade)
- **arestas**: as relações entre essas entidades

Diferente de uma tabela tradicional, o grafo foi feito para responder perguntas relacionais com profundidade, como "quem está conectado com o quê" e "por qual caminho".

Por isso ele é tão poderoso: quando os dados são altamente conectados, consultar caminhos e vizinhanças vira algo natural e explicável.

A menor unidade de conhecimento aqui é a tripla:

- `subject`
- `relation`
- `object`

Com tipos e confiança, por exemplo:

- `Ana (User) -> reported_issue -> cheiro ruim (Issue)`
- `cheiro ruim (Issue) -> affects_location -> quarto 2 (Location)`
- `Ana (User) -> requested_action -> reembolso parcial (Activity)`

Quando juntamos milhares dessas triplas, formamos um grafo de conhecimento.

Por que isso importa?

- dá para consultar recorrência,
- dá para explicar conexões via caminhos,
- dá para rastrear de qual mensagem cada relação veio.

## TODO: Exemplo das conversas

---

## Pipeline de agentes que extrai as triplas

A pipeline atual do backend segue esta ordem:

1. `extraction_agent` extrai triplas da conversa recente.
2. Entra uma camada de `domain policy` para reforçar relações obrigatórias por domínio.
3. Fazemos canonicalização local (normalização de nomes e relações).
4. Resolvemos entidades (reuso de entidades existentes) com `resolution_agent` ou atalho fuzzy local.
5. `policy_agent` aplica gate semântico para evitar ruído ontológico.
6. Dedupe semântico e upsert no Neo4j.

No fim, cada relação salva metadados de rastreio (mensagem de origem, confidence, timestamps e contador de menções).

---

## Como o chatbot admin caminha o grafo
TODO: adicionar prints do adm chat

No modo admin, o assistente usa tools para explorar o grafo com segurança.

Tools principais:

- `describe_graph_schema`
- `find_entity`
- `neighbors`
- `shortest_path`
- `graph_stats`
- `recent_relations`
- `run_graph_query` (somente leitura)

Exemplos de perguntas que funcionam bem:

- "Quais problemas mais recorrentes por quarto?"
- "Quais clientes pediram reembolso?"
- "Qual o menor caminho entre Diego e quarto 7?"

Exemplo de caminho explicável:

- `Diego -> reported_issue -> troca de enxoval -> affects_location -> quarto 7`

Esse tipo de resposta é importante porque não apenas diz o resultado, mas mostra a trilha de evidências no grafo.

---

## Outras indústrias onde isso encaixa muito bem

### E-commerce

Mesmo princípio, outro domínio:

- Usuário interessado em produto
- Comparação com concorrente
- Problema de entrega/pagamento
- Pedido de ação (troca, cancelamento, reembolso)

Com o perfil de prompt certo, dá para mapear:

- produtos com maior intenção de compra,
- concorrentes mais citados,
- gargalos de experiência por etapa,
- padrões por segmento de cliente.

Outros cenários naturais: SaaS support, telecom, saúde, educação e suporte financeiro.

---

## Aprendizados: Importancia do conhecimento de domínio

Um aprendizado forte foi: sem contexto de domínio, a IA pode criar conexões que não importam para ti.

Se você não explica com clareza "que tipo de coisa deve entrar no grafo", o LLM mistura:

- fatos concretos (bons), com
- artefatos de processo ou interpretações vagas (ruído).

### Domain policy na prática

Domain policy é o contrato semântico do seu grafo.

Exemplo no hotel:

- `User -> reported_issue -> Issue`
- `Issue -> affects_location -> Location`
- `User -> requested_action -> Activity`

Com isso, o pipeline ganha previsibilidade e o grafo fica muito mais útil para consulta analítica.

Sem isso, você até extrai triplas, mas perde consistência entre sessões e as queries quebram facilmente.

---

## Como a resolução de entidades é feita hoje

Hoje usamos uma abordagem híbrida e pragmática:

1. Busca de candidatos no grafo (`find_entity`) por nome e tokens relevantes.
2. Score local combinando similaridade de string e sobreposição de tokens.
3. Reuso da entidade existente quando o score passa o threshold por tipo.
4. Em casos mais ambíguos, `resolution_agent` usa tools adicionais para decidir.

Funciona bem para começar e é simples de operar.

Limitações atuais:

- depende bastante de similaridade lexical,
- sofre mais com sinônimos e paráfrases distantes,
- requer ajustes finos de threshold por tipo de entidade.

---

## Melhorias futuras: embeddings para resolver entidades e relações

Uma evolução natural é adicionar resolução semântica com embeddings.

Ideia de arquitetura:

1. Gerar embedding para entidades candidatas e para novas menções.
2. Buscar vizinhos mais próximos em um índice vetorial.
3. Re-rank com regras de domínio (tipo de entidade, contexto local, relações existentes).
4. Confirmar merge/reuso com confiança calibrada.

Ganhos esperados:

- melhor tratamento de sinônimos e variações linguísticas,
- menos duplicação semântica,
- menor dependência de heurísticas de string matching.

Extensão futura: usar embeddings também para sugerir relações prováveis (sempre com policy gate para evitar alucinação estrutural).

---

## Conclusão

O ponto principal não é apenas ter um chatbot que responde bem.

O diferencial está em transformar conversa em estrutura consultável, com qualidade semântica e rastreabilidade.

No caso de hotel, isso significa:

- enxergar recorrência por local,
- conectar experiência do hóspede com impacto operacional,
- e responder perguntas analíticas com evidências no grafo.

Em resumo: memória textual ajuda. Memória estruturada possibilita insights de negócio previamente impossíveis.
