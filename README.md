# Worker de Monitoramento de Aves

Worker Python (Poetry) que roda continuamente e:

1. lista somente as fotos na **raiz** do bucket S3 `comedouro-aves-fotos-tcc-2026`;
2. classifica cada foto com um modelo Keras/InceptionV3 treinado localmente (`model.keras`);
3. quando prevê uma das três espécies ou `Outros` com confiança igual ou superior a
   `CONFIDENCE_THRESHOLD` (padrão `0.85`), cria um avistamento autenticado na API Aves;
   espécies usam sua pasta e `Outros` usa `outros/`, aparecendo como **Ave não identificada**;
4. quando prevê `Background`, move a foto para `background/` sem criar avistamento,
   independentemente da confiança. Qualquer previsão abaixo do limiar, incluindo
   `Outros`, também vai para `background/`, sem aparecer no app;
5. quando o arquivo não pode ser decodificado como imagem, move para `erros/` (nunca apaga).

O worker é seguro para reiniciar a qualquer momento: o estado de cada foto (chave S3 + ETag)
é persistido em SQLite local, e a criação de avistamento usa uma chave de idempotência
determinística (`sha256(bucket:key:etag)`) reconhecida pela API — retomar após uma queda nunca
duplica um avistamento.

## Requisitos

- Windows 11 (ou qualquer SO com Python nativo), **Python 3.12** (`>=3.12,<3.13`).
- [Poetry](https://python-poetry.org/) instalado (`pipx install poetry` ou instalador oficial).
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) para configurar o AWS SSO.
- Um modelo `.keras` treinado (InceptionV3, entrada RGB 299×299, `preprocess_input`
  incorporado no grafo, pixels de entrada em 0–255). O modelo novo possui **cinco**
  saídas softmax na ordem `Avoante`, `Bem-Te-Vi`, `Sabia-Laranjeira`, `Outros`, `Background`;
  mantenha o `class_names.json` exportado pelo notebook no **mesmo diretório** do modelo
  para que o worker valide a ordem antes de iniciar. Modelos antigos de três saídas
  continuam aceitos sem esse arquivo; modelos de quatro saídas continuam aceitos
  com seus rótulos, mas não distinguem `Background`. Os arquivos do modelo
  **não são versionados**.

Exemplo: copie `inceptionv3_classificador_passaros.keras` e `class_names.json` para
`models/`, e configure `MODEL_PATH=./models/inceptionv3_classificador_passaros.keras`.
O catálogo local mantém **três espécies identificadas** em `species` e uma categoria
separada em `unidentifiedBird`. O comando `seed-species` registra essa categoria na
API como **Ave não identificada**. O campo obrigatório `scientificName` usa o marcador
**Espécie não identificada**, não um nome científico inferido. `Background` nunca é
cadastrado. A categoria não exige uma imagem de catálogo: os avistamentos usam a
própria foto, e o app já oferece um estado sem imagem para o catálogo.

`Outros` representa aves fora das três espécies identificadas; `Background`
representa cenas sem aves. Avistamentos de `Outros` usam a mesma outbox e geram
histórico/push pelas regras da API. Fotos rejeitadas não criam trabalho de notificação.
Ao migrar, pare o worker, separe as fotos vazias do antigo dataset `Outros` em
`Background`, treine as cinco classes e substitua modelo/rótulos juntos. Um modelo
antigo de quatro classes pode reconhecer cenas vazias como `Outros` e publicá-las;
não o use para validar a separação nova. Execute `seed-species` antes de iniciar
o worker atualizado; a ausência da categoria na API interrompe a inicialização
com orientação explícita, sem processar fotos.
O `threshold_report.json` do notebook é exploratório; o worker **não** lê nem aplica
automaticamente um limiar dele. Revise os erros e a cobertura antes de ajustar
`CONFIDENCE_THRESHOLD` e ativar o processamento real.

## Instalação

```powershell
cd Worker_monitoramentoAves
poetry install
```

Isso cria um virtualenv local e instala todas as dependências, incluindo TensorFlow — a
primeira instalação pode demorar vários minutos.

## Configuração de credenciais AWS (SSO)

O worker nunca deve usar chaves de acesso estáticas. Configure um perfil do AWS IAM Identity
Center (SSO) uma única vez:

```powershell
aws configure sso
# Guarde o nome do perfil configurado, ex.: aves-worker
aws sso login --profile aves-worker
```

A sessão SSO expira periodicamente; quando expirar, o worker falhará com uma mensagem clara e
**preservará todas as fotos na raiz** (nada é apagado/movido em caso de falha de credencial).
Basta rodar `aws sso login --profile <perfil>` novamente e reiniciar o worker.

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha os valores (o arquivo `.env` nunca deve ser
versionado nem compartilhado — restrinja sua permissão de leitura ao usuário que executa o
worker):

```powershell
copy .env.example .env
```

Principais variáveis (veja `.env.example` para a lista completa e os valores recomendados):

| Variável | Descrição |
|---|---|
| `AWS_PROFILE` | Nome do perfil SSO configurado acima. |
| `AWS_REGION` | Região do bucket/API (`us-east-1`). |
| `S3_BUCKET` | Nome do bucket do comedouro. |
| `API_BASE_URL` | URL base da API Aves implantada. |
| `API_USER_NAME` / `API_USER_EMAIL` / `API_USER_PASSWORD` | Credenciais da conta de serviço dedicada ao worker. |
| `MODEL_PATH` | Caminho local do modelo `.keras` (para 4 ou 5 saídas, `class_names.json` deve estar na mesma pasta). |
| `STATE_DB_PATH` | Caminho do arquivo SQLite de estado (criado automaticamente). |
| `POLL_INTERVAL_SECONDS` | Intervalo entre ciclos de verificação (padrão `60`). |
| `MAX_IMAGES_PER_CYCLE` | Máximo de fotos processadas por ciclo (padrão `10`). |
| `NOTIFICATION_PROCESSING_ENABLED` | Opt-in para chamar o executor restrito da API (padrão `false`). Nunca processa em dry-run. |
| `CONFIDENCE_THRESHOLD` | Confiança mínima para publicar uma espécie ou `Outros` (padrão `0.85`); `Background` nunca é publicado. |
| `CAPTURE_TIMEZONE` | Fuso horário de interpretação dos timestamps (padrão `America/Sao_Paulo`). |

## Primeira execução (bootstrap)

Execute nesta ordem, uma única vez (ambos os comandos são seguros para reexecução):

```powershell
poetry run feeder-worker bootstrap-service-user
poetry run feeder-worker seed-species
```

- `bootstrap-service-user` cria a conta de serviço na API (ou confirma que já existe) e valida
  o login, sem persistir a senha ou o token.
- `seed-species` garante que as três espécies do catálogo (`Avoante`, `Bem-Te-Vi`,
  `Sabia-Laranjeira`) e a categoria **Ave não identificada** existam na API,
  sem sobrescrever conteúdo já curado. A categoria permite exibir `Outros`
  sem afirmar uma identificação taxonômica; `Background` nunca é cadastrado.

## Ensaio (dry-run)

Antes de rodar de verdade, valide a classificação sem criar avistamentos, mover ou apagar nada:

```powershell
poetry run feeder-worker run --dry-run --once
```

Revise a saída (rótulo/confiança previstos e a ação planejada para cada foto) antes de prosseguir,
especialmente as fotos classificadas como `Outros` e `Background`. O modelo de cinco classes não garante
uma taxa de falsos positivos: o conjunto de validação do notebook tem poucos exemplos de
`Background`, e `CONFIDENCE_THRESHOLD=0.85` é apenas um valor inicial.

## Execução contínua

```powershell
poetry run feeder-worker run
```

O worker roda em loop até ser interrompido (Ctrl+C ou `--once` para um único ciclo).

### Notificações pendentes

Após implantação/ativação autorizada da API, configure
`NOTIFICATION_PROCESSING_ENABLED=true`. A API deve ter o mesmo flag habilitado
e `NOTIFICATION_RUNNER_USER_ID` igual ao UUID da conta de serviço existente.
Outra conta recebe 403; processamento desabilitado ou sem configuração
retorna 503. O client renova JWT uma vez após 401, conforme o login existente.

Antes e depois de cada ciclo, inclusive sem fotos e em `--once`, o worker
chama `POST /api/internal/notification-jobs/process`, sem parâmetros de envio.
Recebe somente contagens de concluídos/adiados/falhas; a chave FCM e tokens
dos dispositivos ficam na API, nunca neste processo. `--dry-run` não chama
o executor nem envia push.

A API grava outbox junto ao avistamento em `aves-sightings`; reiniciar o
worker não perde trabalhos. Falha do executor é registrada sem reabrir fotos
concluídas ou interromper classificação. Worker parado, bootstrap/modelo
inválido ou credenciais indisponíveis atrasam histórico/push até retomar.
O intervalo de 60 s não é SLA de entrega. Falhas terminais ficam persistidas
na API para revisão autorizada, sem apagar histórico.

No rollback, desligue este flag e o processamento da API, preservando SQLite,
fila e caixas. Ativação real exige confirmação e APK Android próprio;
testes locais não enviam push.

## Implantação como serviço no Windows (Agendador de Tarefas)

1. Use o script `scripts/run_worker.ps1` (fixa o diretório de trabalho na raiz do projeto,
   executa `poetry run feeder-worker run` e grava a saída em `logs/worker_<data>.log`, já que o
   Agendador de Tarefas não exibe o console) como ação da tarefa agendada. Aceita `-DryRun` e
   `-Once` para testes manuais (`powershell -File scripts\run_worker.ps1 -DryRun -Once`).
2. Crie uma tarefa no Agendador de Tarefas do Windows:
   - Gatilho: "Ao iniciar o computador" (ou logon do usuário de serviço).
   - Ação: executar `powershell.exe -ExecutionPolicy Bypass -File scripts\run_worker.ps1`.
   - Em "Configurações", habilite "Reiniciar a tarefa se ela falhar" com poucos minutos de
     intervalo, e "Executar a tarefa assim que possível após uma inicialização agendada perdida".
3. Como o Agendador de Tarefas pode forçar o encerramento do processo (`TerminateProcess`, que
   nenhum tratamento de sinal consegue interceptar), a segurança contra duplicidade depende do
   design de idempotência descrito acima, não do encerramento gracioso — está tudo coberto mesmo
   em um encerramento abrupto.
4. Lembre-se de que a sessão AWS SSO expira periodicamente e precisa de renovação interativa
   (`aws sso login --profile <perfil>`); monitore os logs do worker para detectar esse caso.

## Testes

```powershell
poetry run pytest
```
