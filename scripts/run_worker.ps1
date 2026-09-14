<#
.SYNOPSIS
    Inicia o worker de monitoramento do comedouro (`feeder-worker run`) de
    forma adequada para o Agendador de Tarefas do Windows.

.DESCRIPTION
    - Fixa o diretório de trabalho na raiz do projeto (independente de onde o
      Agendador de Tarefas inicie o processo).
    - Registra saída em um arquivo de log com rotação simples por data, já
      que o Agendador de Tarefas não exibe a saída do console.
    - Falha rápido e com mensagem clara se o Poetry ou o ambiente virtual não
      estiverem prontos, em vez de falhar silenciosamente.
    - Repassa o código de saída do worker para o Agendador de Tarefas, para
      que a política de reinício em caso de falha funcione corretamente.

.NOTES
    Uso esperado como Ação de uma Tarefa Agendada:
        powershell.exe -ExecutionPolicy Bypass -File scripts\run_worker.ps1
#>

[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$Once
)

$ErrorActionPreference = "Stop"

# Raiz do projeto = diretório pai deste script (scripts/..).
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ("worker_{0}.log" -f (Get-Date -Format "yyyy-MM-dd"))

function Write-Log {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $LogFile -Value $line
    Write-Output $line
}

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    Write-Log "ERRO: 'poetry' não foi encontrado no PATH. Instale o Poetry e reinicie a tarefa."
    exit 1
}

$feederArgs = @("run")
if ($DryRun) { $feederArgs += "--dry-run" }
if ($Once) { $feederArgs += "--once" }

Write-Log "Iniciando feeder-worker $($feederArgs -join ' ') em $ProjectRoot"

try {
    poetry run feeder-worker @feederArgs *>> $LogFile
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Log "ERRO ao executar feeder-worker: $($_.Exception.Message)"
    exit 1
}

Write-Log "feeder-worker encerrou com código de saída $exitCode"
exit $exitCode
