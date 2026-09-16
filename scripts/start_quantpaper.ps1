[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [string]$ConfigPath = "",
    [string]$DbPath = "",
    [switch]$NoBrowser,
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimePath = Join-Path $projectRoot "data\runtime"
$logPath = Join-Path $projectRoot "data\logs"
$portSuffix = if ($Port -eq 8000) { "" } else { "-$Port" }
$apiPidPath = Join-Path $runtimePath "quantpaper-api$portSuffix.pid"
$runnerPidPath = Join-Path $runtimePath "quantpaper-runner$portSuffix.pid"
$oldConfig = [Environment]::GetEnvironmentVariable("QUANTPAPER_CONFIG", "Process")
$oldDbPath = [Environment]::GetEnvironmentVariable("QUANTPAPER_DB_PATH", "Process")

function Restore-ProcessEnvironment {
    if ($null -eq $oldConfig) {
        Remove-Item Env:QUANTPAPER_CONFIG -ErrorAction SilentlyContinue
    } else {
        $env:QUANTPAPER_CONFIG = $oldConfig
    }
    if ($null -eq $oldDbPath) {
        Remove-Item Env:QUANTPAPER_DB_PATH -ErrorAction SilentlyContinue
    } else {
        $env:QUANTPAPER_DB_PATH = $oldDbPath
    }
}

function Show-LauncherError {
    param([string]$Message)

    try {
        Add-Type -AssemblyName System.Windows.Forms
        [System.Windows.Forms.MessageBox]::Show(
            $Message,
            "QuantPaper startup failed",
            [System.Windows.Forms.MessageBoxButtons]::OK,
            [System.Windows.Forms.MessageBoxIcon]::Error
        ) | Out-Null
    } catch {
        Write-Error $Message
    }
}

function Test-ApiReady {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-ManagedProcess {
    param(
        [int]$ProcessId,
        [string]$CommandName
    )

    if ($ProcessId -le 0) {
        return $false
    }
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $process) {
        return $false
    }
    $pattern = "quantpaper\.cli\s+$([regex]::Escape($CommandName))"
    return [bool]($process.CommandLine -match $pattern)
}

function Find-ManagedProcess {
    param([string]$CommandName)

    $pattern = "quantpaper\.cli\s+$([regex]::Escape($CommandName))"
    $processes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue)
    foreach ($process in $processes) {
        if ($process.CommandLine -and $process.CommandLine -match $pattern) {
            return [int]$process.ProcessId
        }
    }
    return 0
}

function Read-ManagedPid {
    param(
        [string]$PidFile,
        [string]$CommandName
    )

    if (Test-Path -LiteralPath $PidFile) {
        try {
            $processId = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
            if (Test-ManagedProcess -ProcessId $processId -CommandName $CommandName) {
                return $processId
            }
        } catch {
        }
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    }
    return Find-ManagedProcess -CommandName $CommandName
}

function Start-ManagedProcess {
    param(
        [string]$CommandName,
        [string]$PidFile,
        [string[]]$Arguments
    )

    $safeName = $CommandName.ToLowerInvariant()
    $stdoutPath = Join-Path $logPath "${safeName}.out.log"
    $stderrPath = Join-Path $logPath "${safeName}.err.log"
    $process = Start-Process `
        -FilePath $pythonPath `
        -ArgumentList $Arguments `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ASCII
    return $process
}

try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Project Python environment was not found: $pythonPath"
    }
    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    New-Item -ItemType Directory -Path $logPath -Force | Out-Null

    if (-not [string]::IsNullOrWhiteSpace($ConfigPath)) {
        $resolvedConfig = Resolve-Path -LiteralPath $ConfigPath -ErrorAction Stop
        $env:QUANTPAPER_CONFIG = $resolvedConfig.Path
    }
    if (-not [string]::IsNullOrWhiteSpace($DbPath)) {
        $dbCandidate = if ([IO.Path]::IsPathRooted($DbPath)) {
            $DbPath
        } else {
            Join-Path $projectRoot $DbPath
        }
        $resolvedDb = [IO.Path]::GetFullPath($dbCandidate)
        $env:QUANTPAPER_DB_PATH = $resolvedDb
    }

    $apiProcessId = Read-ManagedPid -PidFile $apiPidPath -CommandName "serve"
    if (-not (Test-ApiReady)) {
        if ($apiProcessId -gt 0) {
            throw "QuantPaper API process exists but health check failed. See data/logs/serve.err.log."
        }
        Start-ManagedProcess `
            -CommandName "serve" `
            -PidFile $apiPidPath `
            -Arguments @("-m", "quantpaper.cli", "serve", "--host", "127.0.0.1", "--port", "$Port") | Out-Null
    }

    $apiReady = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        if (Test-ApiReady) {
            $apiReady = $true
            break
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $apiReady) {
        throw "QuantPaper API did not become ready. See data/logs/serve.err.log."
    }

    $runnerProcessId = Read-ManagedPid -PidFile $runnerPidPath -CommandName "runner"
    if ($runnerProcessId -le 0) {
        Start-ManagedProcess `
            -CommandName "runner" `
            -PidFile $runnerPidPath `
            -Arguments @("-m", "quantpaper.cli", "runner") | Out-Null
    }

    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:$Port/" | Out-Null
    }
} catch {
    if ($Quiet) {
        Write-Error $_.Exception.Message
    } else {
        Show-LauncherError -Message $_.Exception.Message
    }
    exit 1
} finally {
    Restore-ProcessEnvironment
}
