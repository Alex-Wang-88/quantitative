[CmdletBinding()]
param(
    [ValidateSet("live", "replay")]
    [string]$Mode = "live",
    [ValidateRange(1, 65535)]
    [int]$Port = 8000,
    [ValidateRange(15, 720)]
    [int]$SessionMinutes = 240,
    [switch]$FreshReplay,
    [switch]$ForceRestart,
    [switch]$NoBrowser,
    [switch]$AllowEastmoneyOutsideWindow
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$powershellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$supervisorPath = Join-Path $PSScriptRoot "quantpaper_supervisor.ps1"
$supervisorTaskName = "QuantPaper Background Runner"
$runtimePath = Join-Path $projectRoot "data\runtime"
$manualSessionPath = Join-Path $runtimePath "manual-session.json"
$replayConfigPath = Join-Path $projectRoot "config\replay.toml"
$replayDbPath = Join-Path $runtimePath "visual-paper.db"

function Stop-ManagedQuantPaper {
    $processes = @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -match "quantpaper\.cli\s+(serve|runner)"
            }
    )
    foreach ($process in $processes) {
        try {
            Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
        } catch {
        }
    }
    Get-ChildItem -LiteralPath $runtimePath -Filter "quantpaper-*.pid" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

function Stop-QuantPaperSupervisor {
    $task = Get-ScheduledTask -TaskName $supervisorTaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        return $false
    }
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $supervisorTaskName -ErrorAction SilentlyContinue
    }
    # Stop-ScheduledTask may only signal a long-running PowerShell action.
    # Remove any already loaded supervisor process before starting the
    # one-shot check, otherwise two launchers can race for the same port.
    foreach ($process in @(
        Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
            Where-Object { $_.CommandLine -match "quantpaper_supervisor\.ps1" }
    )) {
        Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
    }
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $remaining = @(
            Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -match "quantpaper_supervisor\.ps1" }
        )
        if ($remaining.Count -eq 0) {
            break
        }
        Start-Sleep -Milliseconds 250
    }
    return $true
}

function Start-QuantPaperSupervisor {
    param([bool]$TaskWasPresent)

    if ($TaskWasPresent) {
        Start-ScheduledTask -TaskName $supervisorTaskName
    }
}

function Wait-ApiReady {
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $response = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw "Project Python environment was not found: $pythonPath"
    }
    if (-not (Test-Path -LiteralPath $supervisorPath)) {
        throw "QuantPaper supervisor was not found: $supervisorPath"
    }
    if ($Mode -eq "replay" -and -not (Test-Path -LiteralPath $replayConfigPath)) {
        throw "Replay configuration was not found: $replayConfigPath"
    }

    New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
    $supervisorTaskWasPresent = Stop-QuantPaperSupervisor
    if ($ForceRestart) {
        Stop-ManagedQuantPaper
    }
    if ($Mode -eq "replay" -and $FreshReplay -and (Test-Path -LiteralPath $replayDbPath)) {
        Remove-Item -LiteralPath $replayDbPath -Force
        Remove-Item -LiteralPath ($replayDbPath + "-shm") -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath ($replayDbPath + "-wal") -Force -ErrorAction SilentlyContinue
    }

    $expiresAt = [DateTimeOffset]::UtcNow.AddMinutes($SessionMinutes)
    $session = [ordered]@{
        mode = $Mode
        port = $Port
        config_path = if ($Mode -eq "replay") { [IO.Path]::GetFullPath($replayConfigPath) } else { "" }
        db_path = if ($Mode -eq "replay") { [IO.Path]::GetFullPath($replayDbPath) } else { "" }
        open_eastmoney = ($Mode -eq "live")
        allow_eastmoney_outside_window = [bool]$AllowEastmoneyOutsideWindow
        opened_at = [DateTimeOffset]::UtcNow.ToString("o")
        expires_at = $expiresAt.ToString("o")
    }
    $session | ConvertTo-Json | Set-Content -LiteralPath $manualSessionPath -Encoding UTF8

    # The supervisor is the single canonical opener.  Calling it once here
    # makes this flow identical to the scheduled/background flow and reuses
    # the logged-in Eastmoney session when live mode is selected.
    & $powershellPath -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File $supervisorPath -Once
    if (-not (Wait-ApiReady)) {
        throw "QuantPaper API did not become ready on port $Port. See data/logs/serve.err.log."
    }
    Start-QuantPaperSupervisor -TaskWasPresent $supervisorTaskWasPresent

    $url = "http://127.0.0.1:$Port/"
    if (-not $NoBrowser) {
        Start-Process $url | Out-Null
    }
    Write-Output ("QuantPaper opened in {0} mode. URL: {1}. Session expires: {2}" -f $Mode, $url, $expiresAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm:ss"))
} catch {
    Remove-Item -LiteralPath $manualSessionPath -Force -ErrorAction SilentlyContinue
    Start-QuantPaperSupervisor -TaskWasPresent $supervisorTaskWasPresent
    Write-Error $_.Exception.Message
    exit 1
}
