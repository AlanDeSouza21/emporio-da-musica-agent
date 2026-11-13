Empório da Música — Agente de Atendimento

Protótipo de um agente de atendimento por texto para a Empório da Música, loja fictícia de instrumentos musicais de Campo Grande/MS. O agente responde sobre catálogo, preços, estoque, promoções, status de pedidos e políticas da loja, usando os CSVs e o manual de políticas fornecidos.

Desenvolvido para o desafio técnico de AI Engineer da Artefact.

Resumo da solução
Linguagem: Python
Modelo: Google Gemini (API gratuita), com fallback automático entre modelos
Abordagem: function calling para os dados (pandas sobre os CSVs) + manual de políticas inserido integralmente no prompt de sistema
Interface: Streamlit (chat) e CLI (usada para gravar os exemplos)
Como rodar
Pré-requisitos
Python 3.10 ou superior
Uma chave da API do Gemini (gratuita, gerada no Google AI Studio)
Passo a passo
bash
# 1. Clonar o repositório
git clone https://github.com/AlanDeSouza21/emporio-da-musica-agent.git
cd emporio-da-musica-agent

# 2. Criar e ativar o ambiente virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar as dependências
pip install -r requirements.txt
Configuração (.env)

### Configuração (.env)

Copie o arquivo de exemplo e preencha sua chave:

```bash
# Windows (PowerShell)
copy .env.example .env
# Linux/macOS
cp .env.example .env
```

Conteúdo do `.env`:

```
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,gemini-3.8-flash
```

- `GEMINI_API_KEY`: sua chave da API do Gemini (obrigatória). Gere uma
  gratuita no [Google AI Studio](https://aistudio.google.com/apikey).
- `GEMINI_MODELS`: lista de modelos em ordem de preferência. Se um modelo
  falhar (indisponível, cota diária esgotada ou aposentado), o agente passa
  para o próximo mantendo o histórico da conversa. Os nomes de modelos do
  Gemini mudam com frequência; confira os disponíveis para a sua chave na
  documentação do Google e ajuste a lista se necessário.
- `TODAY_OVERRIDE` (opcional, formato AAAA-MM-DD): força a data "de hoje"
  usada pelo agente. Veja a seção "Suposições assumidas".

### Executar
bash
# Interface web
streamlit run app.py

# Alternativa: chat no terminal (grava a sessão em examples/)
python chat_cli.py

Os dados (data/) já estão no repositório. PDF foi renomeado para politicas_da_loja.pdf, sem acento, para simplificar caminhos.

Estrutura do projeto
.
├── app.py                  # Interface Streamlit
├── chat_cli.py             # Chat no terminal (também grava as conversas de exemplo)
├── requirements.txt        # serve para mostrar as bibliotecas que foram instaladas
├── data/                   # CSVs e PDF de políticas
├── examples/               # Conversas de exemplo
└── src/
    ├── viewer.ipynb        # serve para visualizar os dados, não tem nenhum efeito prático no projeto e se for retirado não afeta o resultado final 
    ├── data_loader.py      # Carga dos CSVs
    ├── data_tools.py       # Ferramentas de consulta e tratamento de dados (produtos, promoções, pedidos)
    └── agent.py            # Prompt, persona, loop de function calling e fallback de modelos
Arquitetura
Cliente (Streamlit/CLI)
        │
        ▼
 Agente (Gemini + prompt de sistema com persona e políticas)
        │  decide se precisa de uma ferramenta
        ├──► search_products / get_product_details / get_active_promotions / list_categories
        ├──► get_order_status (exige número do pedido + e-mail)
        │            └─ pandas sobre os CSVs
        └──► resposta direta (políticas no prompt, saudações, fora do escopo)
A mensagem do cliente entra no agente junto com o histórico da conversa.
O modelo decide se responde direto ou se chama uma ferramenta.
Quando chama, o código executa a função Python correspondente, devolve o resultado ao modelo e repete o ciclo (no máximo 6 rodadas) até haver uma resposta final.
O modelo escreve a resposta final na persona da loja.
Decisões técnicas e justificativas
Abordagem: function calling + contexto para políticas (híbrido)

O enunciado pede que o agente saiba quando consultar dados e quando consultar políticas:

Dados (preço, estoque, promoções, pedidos): são tabelas e precisam de resultado exato. Por isso são consultados por funções Python com pandas, expostas ao modelo como ferramentas (function calling), e não por embeddings. Busca semântica poderia devolver um produto parecido com um preço diferente; uma consulta determinística não.
Políticas (trocas, devoluções, horários, pagamento): o manual é curto, então o texto extraído do PDF (pypdf) é inserido inteiro no prompt de sistema. O modelo consulta as políticas diretamente, sem etapa de recuperação.
Por que não usei RAG para as políticas

Possíveis melhorias
Meu plano inicial era usar RAG para as políticas (chunks por seção, embeddings e banco vetorial). Com o prazo de entrega, priorizei ter o agente completo funcionando de ponta a ponta e optei por inserir o texto do manual inteiro no prompt de sistema.

Modelo e provedor: Gemini (gratuito) com fallback

Escolhi a API gratuita do Google Gemini por não ter custo e por suportar function calling. Na prática, o plano gratuito trouxe três problemas, que moldaram o código:

Cota diária por modelo (cerca de 20 requisições/dia por modelo no teste). Cada mensagem do cliente gasta mais de uma requisição (uma para o pedido de ferramenta e outra para a resposta final).
Modelos aposentados (erro 404) e picos de demanda (erro 503).
Alguns modelos "lite" não acionaram a execução automática de ferramentas do SDK.

Por isso o agente: (a) executa o loop de ferramentas manualmente, o que funciona igual em qualquer modelo e deixa o registro das chamadas confiável; (b) tenta de novo com espera crescente em erros temporários (500/503/504); e (c) passa para o próximo modelo da lista em erros 404, 429 e 503, mantendo o histórico.

Tratamento dos dados

Feito em data_tools.py, na carga:

Normalização para busca: nomes de produto e categoria em minúsculas, sem acentos e com plural simplificado (ex.: "violões" e "violao" encontram o mesmo produto).
Promoções: só as ativas (is_active = 1) entram; o preço promocional é calculado a partir do percentual (price_brl × (1 − desconto)), e a busca por orçamento considera o preço promocional.
Produtos: a busca de catálogo considera apenas produtos com status active; produtos esgotados são ocultados por padrão.
Especificações: o campo specs, que vem como texto JSON, é convertido em dicionário.
Valores ausentes: NaN vira None antes de ir para o modelo.
Junções: orders + order_items + products + customers são combinadas na consulta de pedidos.
Estratégia de prompt

O prompt de sistema define: persona (atendente simpático e objetivo, em português do Brasil), regras de uso de ferramentas (nunca inventar preço, estoque ou dados de pedido; usar exatamente o total_found; sempre passar o orçamento como max_price), regras de política (responder só com base no manual e indicar a equipe quando a informação não existir), tratamento de assuntos fora do escopo (recusa educada com redirecionamento) e resistência a tentativas de mudar o papel do agente. A data atual também é injetada, para que regras com prazo (como a devolução) possam ser aplicadas.

Privacidade nos pedidos

get_order_status exige número do pedido e e-mail do cliente. Se o e-mail não corresponde, a função devolve o mesmo erro de "pedido não encontrado", sem indicar qual dos dois estava errado. O prompt também instrui o agente a pedir os dois dados antes de chamar a ferramenta.

Persistência e interface

O histórico fica em st.session_state (por sessão do navegador) e dentro do chat do modelo. Não há persistência em disco. A interface é Streamlit por ser simples e permitir demonstrar o agente com pouco código. O chat_cli.py existe para testes rápidos e grava cada sessão em examples/.

Suposições assumidas
Disponibilidade: "disponível" significa produto ativo com estoque maior que zero.
Identificação do cliente: o e-mail cadastrado é suficiente para liberar dados de um pedido (é um protótipo, sem autenticação real).
Data de "hoje": o agente usa a data real do sistema. Os pedidos de exemplo são de 2025, então quase toda devolução por arrependimento aparece como fora do prazo. Para demonstrar um caso dentro do prazo, defina TODAY_OVERRIDE com uma data próxima da entrega do pedido (por exemplo, TODAY_OVERRIDE=2025-10-28 para o pedido 1, entregue em 25/10/2025).
Ações: o agente não processa devoluções, pagamentos ou alterações; explica o procedimento das políticas e direciona o cliente à equipe.
Resultados de busca: a ferramenta devolve no máximo 8 produtos por consulta, informando o total encontrado.
Exemplos de interação

Em examples/ há conversas reais com o agente, gravadas pelo chat_cli.py (cada resposta indica as ferramentas chamadas e seus argumentos):

Catálogo com orçamento: violões disponíveis até R$ 1.000
Consulta de preço: Takamine GD20
Informações da loja: endereço e horário (respondidas pelas políticas)
Devolução: pedido 1, cruzando a data de entrega com o prazo de arrependimento da política (caso não trivial: usa dados do pedido e regra do manual)
Pergunta fora do escopo
Limitações conhecidas
Cota e estabilidade do plano gratuito: limite diário por modelo e erros 503 ocasionais. O fallback ameniza, mas todos os modelos da lista podem falhar.
Busca de produtos por palavras-chave: a busca é textual sobre nome e categoria normalizados. Não entende sinônimos ("violão de aço" vs. "folk") nem erros de digitação.
Políticas inteiras no prompt: não escala para manuais grandes e consome mais tokens por requisição.
Erros do modelo: durante os testes, o modelo chegou a omitir o filtro de preço em uma busca. Reforcei o prompt e passei a devolver os filtros aplicados (filters_applied) no resultado, mas o comportamento depende do modelo e não há uma avaliação automatizada.
Sem testes automatizados: a validação foi manual, por conversas.
Estado compartilhado: o registro de ferramentas chamadas é uma lista em nível de módulo, o que não é seguro para vários usuários simultâneos em produção.
Sem persistência do histórico entre sessões.
Com mais tempo
Testes automatizados das ferramentas de dados e uma suíte de avaliação do agente (perguntas esperadas × ferramentas e respostas esperadas).
RAG para as políticas (chunks por seção, embeddings, banco vetorial), caso o manual cresça.
Busca de produtos mais robusta (tolerância a erros de digitação e busca semântica nas descrições).
Persistência do histórico (SQLite) e identificação do cliente mais forte.
Encaminhamento para atendimento humano com resumo da conversa.
Registro estruturado (logs) e métricas de custo e latência por conversa.
Uso de assistentes de código

Usei o Claude (Anthropic), pelo chat, como assistente durante todo o desenvolvimento:

Planejamento: discuti a arquitetura (function calling para dados, políticas no contexto), a ordem de implementação e o que cortar para caber no prazo.
Código: o Claude escreveu as versões iniciais de data_tools.py, agent.py e app.py, que eu executei, testei e ajustei.
Depuração: colei os erros que apareceram (conexão fechada do cliente, erros 429, 503 e 404 por cota e modelos aposentados, resposta vazia por falta de execução das ferramentas) e apliquei as correções sugeridas.
Validação: conferi as respostas do agente contra os CSVs e o PDF. Foi assim que encontrei o problema do filtro de preço omitido e a contagem de produtos errada, corrigidos com ajustes no prompt e no resultado da ferramenta.
Decisões: a escolha do modelo, as suposições e o escopo final foram minhas, com base nas limitações que encontrei nos testes.