# Worker de Monitoramento de Aves

Worker Python (Poetry) que roda continuamente e:

1. lista somente as fotos na **raiz** do bucket S3 `comedouro-aves-fotos-tcc-2026`;
2. classifica cada foto com um modelo Keras/InceptionV3 treinado localmente (`model.keras`);
3. quando a confiança atinge `CONFIDENCE_THRESHOLD` (padrão `0.85`), cria um avistamento
   autenticado na API Aves e move a foto para a pasta da espécie reconhecida;
4. quando a foto é válida mas nenhuma espécie é reconhecida, remove a foto da raiz;
5. quando o arquivo não pode ser decodificado como imagem, move para `erros/` (nunca apaga).

O worker é seguro para reiniciar a qualquer momento: o estado de cada foto (chave S3 + ETag)
é persistido em SQLite local, e a criação de avistamento usa uma chave de idempotência
determinística (`sha256(bucket:key:etag)`) reconhecida pela API — retomar após uma queda nunca
duplica um avistamento.

## Requisitos

- Windows 11 (ou qualquer SO com Python nativo), **Python 3.12** (`>=3.12,<3.13`).
- [Poetry](https://python-poetry.org/) instalado (`pipx install poetry` ou instalador oficial).
- [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) para configurar o AWS SSO.
- Um arquivo `model.keras` treinado (InceptionV3, entrada RGB 299×299, `preprocess_input`
  incorporado no grafo do modelo, saída softmax de 3 classes na ordem
  `Avoante`, `Bem-Te-Vi`, `Sabia-Laranjeira`). **Não é versionado neste repositório.**

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
| `MODEL_PATH` | Caminho absoluto local para `model.keras`. |
| `STATE_DB_PATH` | Caminho do arquivo SQLite de estado (criado automaticamente). |
| `POLL_INTERVAL_SECONDS` | Intervalo entre ciclos de verificação (padrão `60`). |
| `MAX_IMAGES_PER_CYCLE` | Máximo de fotos processadas por ciclo (padrão `10`). |
| `CONFIDENCE_THRESHOLD` | Confiança mínima para reconhecer uma espécie (padrão `0.85`). |
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
  `Sabia-Laranjeira`) existam na API, sem sobrescrever conteúdo já curado.

## Ensaio (dry-run)

Antes de rodar de verdade, valide a classificação sem criar avistamentos, mover ou apagar nada:

```powershell
poetry run feeder-worker run --dry-run --once
```

Revise a saída (rótulo/confiança previstos e a ação planejada para cada foto) antes de prosseguir.

## Execução contínua

```powershell
poetry run feeder-worker run
```

O worker roda em loop até ser interrompido (Ctrl+C ou `--once` para um único ciclo).

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
