[CmdletBinding()]
param(
    [int]$PollSeconds = 60,
    [switch]$Once
)

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcherPath = Join-Path $PSScriptRoot "start_quantpaper.ps1"
$calendarUpdaterPath = Join-Path $PSScriptRoot "update_trading_calendar.py"
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimePath = Join-Path $projectRoot "data\runtime"
$logPath = Join-Path $projectRoot "data\logs\quantpaper-supervisor.log"
$calendarPath = Join-Path $runtimePath "trading_calendar.json"
$eastmoneyOwnershipPath = Join-Path $runtimePath "eastmoney-supervisor-owned.json"
$eastmoneyClientOwnershipPath = Join-Path $runtimePath "eastmoney-client-supervisor-owned.json"
$eastmoneyAdapterPidPath = Join-Path $runtimePath "eastmoney-terminal-adapter.pid"
$eastmoneyAdapterOutPath = Join-Path $projectRoot "data\logs\eastmoney-terminal-adapter.out.log"
$eastmoneyAdapterErrPath = Join-Path $projectRoot "data\logs\eastmoney-terminal-adapter.err.log"
$manualSessionPath = Join-Path $runtimePath "manual-session.json"
$apiPidPath = Join-Path $runtimePath "quantpaper-api.pid"
$runnerPidPath = Join-Path $runtimePath "quantpaper-runner.pid"
$powershellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$activeStartMinutes = 550  # 09:10, leaves a short warm-up before continuous trading
$activeEndMinutes = 930    # 15:30, after the regular session
$eastmoneyPath = if ($env:QUANTPAPER_EASTMONEY_TERMINAL) {
    $env:QUANTPAPER_EASTMONEY_TERMINAL
} else {
    "C:\eastmoney\dfcf\EastMoneyGoldminer\goldminer3\emgm3.exe"
}
$eastmoneyClientPath = if ($env:QUANTPAPER_EASTMONEY_CLIENT) {
    $env:QUANTPAPER_EASTMONEY_CLIENT
} else {
    "C:\eastmoney\dfcf\mainfree.exe"
}
$eastmoneyClientIconPath = "C:\eastmoney\dfcf\res\Main.ico"
$eastmoneyStarterPath = if ($env:QUANTPAPER_EASTMONEY_STARTER) {
    $env:QUANTPAPER_EASTMONEY_STARTER
} else {
    "C:\eastmoney\dfcf\EastMoneyGoldminer\gmstarter.exe"
}
$eastmoneyTokenVariable = "QUANTPAPER_EASTMONEY_TOKEN"
$eastmoneySdkPythonPath = Join-Path $projectRoot ".venv-eastmoney\Scripts\python.exe"
$eastmoneyAdapterScriptPath = Join-Path $PSScriptRoot "eastmoney_terminal_adapter.py"

New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path -Parent $logPath) -Force | Out-Null

function Write-SupervisorLog {
    param([string]$Message)

    $line = "{0} {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line -Encoding UTF8
}

function Get-ShanghaiNow {
    return [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTimeOffset]::UtcNow, "China Standard Time")
}

function Get-CalendarState {
    if (-not (Test-Path -LiteralPath $calendarPath)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $calendarPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-SupervisorLog "Trading calendar cache is invalid: $($_.Exception.Message)"
        return $null
    }
}

function Get-ManualSession {
    if (-not (Test-Path -LiteralPath $manualSessionPath)) {
        return $null
    }
    try {
        $session = Get-Content -LiteralPath $manualSessionPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $bootTime = $null
        try {
            $bootTime = [DateTimeOffset](Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime
        } catch {
            Write-SupervisorLog "Unable to read Windows boot time while validating the manual session."
        }
        if ($bootTime -and $session.opened_at) {
            $openedAt = [DateTimeOffset]::Parse([string]$session.opened_at)
            if ($openedAt -lt $bootTime) {
                Write-SupervisorLog "Manual QuantPaper session predates the current Windows boot; removing stale marker."
                Remove-Item -LiteralPath $manualSessionPath -Force -ErrorAction SilentlyContinue
                return $null
            }
        }
        if (-not $session.expires_at) {
            return $session
        }
        $expiresAt = [DateTimeOffset]::Parse([string]$session.expires_at)
        if ($expiresAt -gt [DateTimeOffset]::UtcNow) {
            return $session
        }
        Write-SupervisorLog "Manual QuantPaper session expired; returning to scheduled mode."
        Remove-Item -LiteralPath $manualSessionPath -Force -ErrorAction SilentlyContinue
    } catch {
        Write-SupervisorLog "Manual QuantPaper session marker is invalid: $($_.Exception.Message)"
        Remove-Item -LiteralPath $manualSessionPath -Force -ErrorAction SilentlyContinue
    }
    return $null
}

function Test-IsTradingDate {
    param([DateTimeOffset]$Now)

    $dateKey = $Now.ToString("yyyy-MM-dd")
    $state = Get-CalendarState
    if ($state -and $state.start -and $state.end) {
        try {
            $start = [DateTime]::ParseExact([string]$state.start, "yyyy-MM-dd", $null)
            $end = [DateTime]::ParseExact([string]$state.end, "yyyy-MM-dd", $null)
            $current = [DateTime]::ParseExact($dateKey, "yyyy-MM-dd", $null)
            if ($current -ge $start -and $current -le $end) {
                return @($state.dates | ForEach-Object { [string]$_ }) -contains $dateKey
            }
        } catch {
            Write-SupervisorLog "Trading calendar cache range is invalid; using weekday fallback."
        }
    }

    # This is only a bootstrap fallback. Once the real provider is available,
    # the cache contains exchange holidays and takes precedence.
    return $Now.DayOfWeek -notin @([DayOfWeek]::Saturday, [DayOfWeek]::Sunday)
}

function Test-InTradingWindow {
    param([DateTimeOffset]$Now)

    $minutes = ($Now.Hour * 60) + $Now.Minute
    return $minutes -ge $activeStartMinutes -and $minutes -lt $activeEndMinutes
}

function Get-EastmoneyProcesses {
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'emgm3.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ExecutablePath -and
                [String]::Equals($_.ExecutablePath, $eastmoneyPath, [StringComparison]::OrdinalIgnoreCase)
            }
    )
}

function Get-EastmoneyMainProcess {
    return @(
        Get-EastmoneyProcesses |
            Where-Object { $_.CommandLine -notmatch "--type=" } |
            Sort-Object ProcessId |
            Select-Object -First 1
    )
}

function Get-EastmoneyClientProcesses {
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'mainfree.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ExecutablePath -and
                [String]::Equals($_.ExecutablePath, $eastmoneyClientPath, [StringComparison]::OrdinalIgnoreCase)
            }
    )
}

function Read-EastmoneyOwnership {
    if (-not (Test-Path -LiteralPath $eastmoneyOwnershipPath)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $eastmoneyOwnershipPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Remove-Item -LiteralPath $eastmoneyOwnershipPath -Force -ErrorAction SilentlyContinue
        return $null
    }
}

function Write-EastmoneyOwnership {
    param([int]$ProcessId)

    @{ pid = $ProcessId; path = $eastmoneyPath; started_at = (Get-ShanghaiNow).ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $eastmoneyOwnershipPath -Encoding UTF8
}

function Read-EastmoneyClientOwnership {
    if (-not (Test-Path -LiteralPath $eastmoneyClientOwnershipPath)) {
        return $null
    }
    try {
        return Get-Content -LiteralPath $eastmoneyClientOwnershipPath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Remove-Item -LiteralPath $eastmoneyClientOwnershipPath -Force -ErrorAction SilentlyContinue
        return $null
    }
}

function Write-EastmoneyClientOwnership {
    param([int]$ProcessId)

    @{ pid = $ProcessId; path = $eastmoneyClientPath; started_at = (Get-ShanghaiNow).ToString("o") } |
        ConvertTo-Json | Set-Content -LiteralPath $eastmoneyClientOwnershipPath -Encoding UTF8
}

function Get-EastmoneyToken {
    $token = [Environment]::GetEnvironmentVariable($eastmoneyTokenVariable, "Process")
    if ([string]::IsNullOrWhiteSpace($token)) {
        $token = [Environment]::GetEnvironmentVariable($eastmoneyTokenVariable, "User")
    }
    return [string]$token
}

function Ensure-EastmoneyClient {
    $existing = @(Get-EastmoneyClientProcesses)
    if ($existing.Count -gt 0) {
        return $true
    }
    if (-not (Test-Path -LiteralPath $eastmoneyClientPath)) {
        Write-SupervisorLog "Eastmoney classic client was not found: $eastmoneyClientPath"
        return $false
    }

    try {
        $process = Start-Process `
            -FilePath $eastmoneyClientPath `
            -ArgumentList @($eastmoneyClientIconPath) `
            -WorkingDirectory (Split-Path -Parent $eastmoneyClientPath) `
            -WindowStyle Normal `
            -PassThru
        $deadline = (Get-Date).AddSeconds(25)
        do {
            Start-Sleep -Seconds 2
            $existing = @(Get-EastmoneyClientProcesses)
        } while ($existing.Count -eq 0 -and (Get-Date) -lt $deadline)
        if ($existing.Count -eq 0) {
            Write-SupervisorLog "Eastmoney classic client launch process PID $($process.Id) exited without a main client process."
            return $false
        }
        Write-EastmoneyClientOwnership -ProcessId ([int]$existing[0].ProcessId)
        Write-SupervisorLog "Started Eastmoney classic client (main PID $($existing[0].ProcessId)); reusing its login session."
        Start-Sleep -Seconds 5
        return $true
    } catch {
        Write-SupervisorLog "Failed to start Eastmoney classic client: $($_.Exception.Message)"
        return $false
    }
}

function Ensure-EastmoneyTerminal {
    $main = @(Get-EastmoneyMainProcess)
    if ($main.Count -gt 0) {
        $ownership = Read-EastmoneyOwnership
        if ($ownership -and ($main.ProcessId -contains [int]$ownership.pid)) {
            return $true
        }
        Write-SupervisorLog "Eastmoney Quant terminal is already open; leaving the existing client unmanaged."
        return $true
    }

    $existing = @(Get-EastmoneyProcesses)
    if ($existing.Count -gt 0) {
        Write-SupervisorLog "Eastmoney renderer processes exist but the main client is not ready; waiting before starting another instance."
        return $false
    }

    if (-not (Test-Path -LiteralPath $eastmoneyPath)) {
        Write-SupervisorLog "Eastmoney Quant terminal was not found: $eastmoneyPath"
        return $false
    }
    if (-not (Ensure-EastmoneyClient)) {
        return $false
    }
    if (-not (Test-Path -LiteralPath $eastmoneyStarterPath)) {
        Write-SupervisorLog "Eastmoney official Quant launcher was not found: $eastmoneyStarterPath"
        return $false
    }

    try {
        $token = Get-EastmoneyToken
        if ([string]::IsNullOrWhiteSpace($token)) {
            Write-SupervisorLog "Eastmoney session token is unavailable; refusing to launch an unauthenticated Quant terminal."
            return $false
        }
        # This is the exact argument shape emitted by the classic client's
        # bottom-left Quant button. A separated '--token', '<value>' form
        # opens the terminal but does not reuse the classic-client session.
        $starterArguments = @("--token=$token")
        $process = Start-Process `
            -FilePath $eastmoneyStarterPath `
            -ArgumentList $starterArguments `
            -WorkingDirectory (Split-Path -Parent $eastmoneyStarterPath) `
            -WindowStyle Normal `
            -PassThru
        $deadline = (Get-Date).AddSeconds(25)
        do {
            Start-Sleep -Seconds 2
            $main = @(Get-EastmoneyMainProcess)
        } while ($main.Count -eq 0 -and (Get-Date) -lt $deadline)
        if ($main.Count -eq 0) {
            Write-SupervisorLog "Eastmoney launch process PID $($process.Id) exited without a visible main client process."
            return $false
        }
        Write-EastmoneyOwnership -ProcessId ([int]$main[0].ProcessId)
        Write-SupervisorLog "Started Eastmoney Quant terminal through the classic-client session (main PID $($main[0].ProcessId)). Waiting for its local service."
        Start-Sleep -Seconds 5
        return $true
    } catch {
        Write-SupervisorLog "Failed to start Eastmoney Quant terminal: $($_.Exception.Message)"
        return $false
    }
}

function Update-TradingCalendarIfNeeded {
    param([DateTimeOffset]$Now)

    if (-not (Test-Path -LiteralPath $pythonPath) -or -not (Test-Path -LiteralPath $calendarUpdaterPath)) {
        return
    }
    $state = Get-CalendarState
    $needsUpdate = $true
    if ($state -and $state.updated_at) {
        try {
            $updatedAt = [DateTimeOffset]::Parse([string]$state.updated_at)
            $needsUpdate = ($Now - $updatedAt).TotalHours -ge 12
        } catch {
            $needsUpdate = $true
        }
    }
    if (-not $needsUpdate) {
        return
    }

    $lastAttempt = $script:lastCalendarAttempt
    if (-not $lastAttempt) {
        $lastAttempt = [DateTimeOffset]::MinValue
    }
    if (($Now - $lastAttempt).TotalMinutes -lt 30) {
        return
    }
    $script:lastCalendarAttempt = $Now
    try {
        $output = & $pythonPath $calendarUpdaterPath 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-SupervisorLog "Trading calendar cache refreshed: $($output -join ' ')"
        } else {
            Write-SupervisorLog "Trading calendar refresh skipped (exit $LASTEXITCODE): $($output -join ' ')"
        }
    } catch {
        Write-SupervisorLog "Trading calendar refresh failed: $($_.Exception.Message)"
    }
}

function Get-ManagedPythonProcesses {
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -match "quantpaper\.cli\s+(serve|runner)"
            }
    )
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    $children = @(
        Get-CimInstance Win32_Process -Filter "ParentProcessId = $ProcessId" -ErrorAction SilentlyContinue
    )
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int]$child.ProcessId)
    }
    try {
        Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
    } catch {
    }
}

function Stop-QuantPaperProcesses {
    foreach ($process in @(Get-ManagedPythonProcesses)) {
        try {
            Stop-ProcessTree -ProcessId ([int]$process.ProcessId)
            Write-SupervisorLog "Stopped QuantPaper process PID $($process.ProcessId) outside the trading window."
        } catch {
            Write-SupervisorLog "Failed to stop QuantPaper PID $($process.ProcessId): $($_.Exception.Message)"
        }
    }
    Get-ChildItem -LiteralPath $runtimePath -Filter "quantpaper-*.pid" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force -ErrorAction SilentlyContinue

    Stop-EastmoneyTerminalAdapter

    $normalizedSdkPython = [IO.Path]::GetFullPath($eastmoneySdkPythonPath)
    foreach ($process in @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ExecutablePath -and
                [String]::Equals($_.ExecutablePath, $normalizedSdkPython, [StringComparison]::OrdinalIgnoreCase) -and
                $_.CommandLine -match "eastmoney_readonly_bridge\.py"
            }
    )) {
        Stop-ProcessTree -ProcessId ([int]$process.ProcessId)
    }
}

function Get-EastmoneyTerminalAdapterProcesses {
    $normalizedSdkPython = [IO.Path]::GetFullPath($eastmoneySdkPythonPath)
    return @(
        Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
            Where-Object {
                $_.ExecutablePath -and
                [String]::Equals($_.ExecutablePath, $normalizedSdkPython, [StringComparison]::OrdinalIgnoreCase) -and
                $_.CommandLine -match "eastmoney_terminal_adapter\.py" -and
                $_.CommandLine -match "--poll"
            }
    )
}

function Ensure-EastmoneyTerminalAdapter {
    if (@(Get-EastmoneyTerminalAdapterProcesses).Count -gt 0) {
        return $true
    }
    if (-not (Test-Path -LiteralPath $eastmoneySdkPythonPath)) {
        Write-SupervisorLog "Eastmoney terminal adapter Python was not found: $eastmoneySdkPythonPath"
        return $false
    }
    if (-not (Test-Path -LiteralPath $eastmoneyAdapterScriptPath)) {
        Write-SupervisorLog "Eastmoney terminal adapter script was not found: $eastmoneyAdapterScriptPath"
        return $false
    }
    try {
        $process = Start-Process `
            -FilePath $eastmoneySdkPythonPath `
            -ArgumentList @($eastmoneyAdapterScriptPath, "--poll", "--interval", "5") `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $eastmoneyAdapterOutPath `
            -RedirectStandardError $eastmoneyAdapterErrPath `
            -PassThru
        Set-Content -LiteralPath $eastmoneyAdapterPidPath -Value $process.Id -Encoding ASCII
        Write-SupervisorLog "Started Eastmoney terminal read-only adapter PID $($process.Id)."
        return $true
    } catch {
        Write-SupervisorLog "Failed to start Eastmoney terminal adapter: $($_.Exception.Message)"
        return $false
    }
}

function Stop-EastmoneyTerminalAdapter {
    foreach ($process in @(Get-EastmoneyTerminalAdapterProcesses)) {
        try {
            Stop-ProcessTree -ProcessId ([int]$process.ProcessId)
            Write-SupervisorLog "Stopped Eastmoney terminal adapter PID $($process.ProcessId) outside the trading window."
        } catch {
            Write-SupervisorLog "Failed to stop Eastmoney terminal adapter PID $($process.ProcessId): $($_.Exception.Message)"
        }
    }
    Remove-Item -LiteralPath $eastmoneyAdapterPidPath -Force -ErrorAction SilentlyContinue
}

function Stop-OwnedEastmoneyTerminal {
    $ownership = Read-EastmoneyOwnership
    if (-not $ownership) {
        return
    }
    foreach ($process in @(Get-EastmoneyProcesses)) {
        try {
            Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction Stop
            Write-SupervisorLog "Stopped supervisor-started Eastmoney Quant terminal PID $($process.ProcessId) outside the trading window."
        } catch {
            Write-SupervisorLog "Failed to stop Eastmoney PID $($process.ProcessId): $($_.Exception.Message)"
        }
    }
    Remove-Item -LiteralPath $eastmoneyOwnershipPath -Force -ErrorAction SilentlyContinue
}

function Stop-OwnedEastmoneyClient {
    $ownership = Read-EastmoneyClientOwnership
    if (-not $ownership) {
        return
    }
    foreach ($process in @(Get-EastmoneyClientProcesses)) {
        try {
            Stop-ProcessTree -ProcessId ([int]$process.ProcessId)
            Write-SupervisorLog "Stopped supervisor-started Eastmoney classic client PID $($process.ProcessId) outside the trading window."
        } catch {
            Write-SupervisorLog "Failed to stop Eastmoney classic client PID $($process.ProcessId): $($_.Exception.Message)"
        }
    }
    Remove-Item -LiteralPath $eastmoneyClientOwnershipPath -Force -ErrorAction SilentlyContinue
}

function Ensure-QuantPaper {
    param([object]$ManualSession = $null)

    try {
        $arguments = @(
            "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden",
            "-File", $launcherPath, "-NoBrowser", "-Quiet"
        )
        if ($ManualSession) {
            if ($ManualSession.config_path) {
                $arguments += @("-ConfigPath", [string]$ManualSession.config_path)
            }
            if ($ManualSession.db_path) {
                $arguments += @("-DbPath", [string]$ManualSession.db_path)
            }
            if ($ManualSession.port) {
                $arguments += @("-Port", [string]$ManualSession.port)
            }
        }
        & $powershellPath @arguments
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            Write-SupervisorLog "QuantPaper launcher exited with code $LASTEXITCODE."
        }
    } catch {
        Write-SupervisorLog "QuantPaper launcher failed: $($_.Exception.Message)"
    }
}

function Invoke-SupervisorCheck {
    $now = Get-ShanghaiNow
    $tradingDate = Test-IsTradingDate -Now $now
    $scheduledActive = $tradingDate -and (Test-InTradingWindow -Now $now)
    $manualSession = Get-ManualSession
    $manualActive = $null -ne $manualSession
    $active = $scheduledActive -or $manualActive

    if ($active) {
        # A manual session may keep QuantPaper available for inspection, but
        # the external Eastmoney clients are only allowed in the trading
        # window.  This prevents a stale/manual marker from opening them at
        # night or on a weekend.
        $openEastmoney = $scheduledActive
        if ($manualSession -and $manualSession.PSObject.Properties.Name -contains "open_eastmoney") {
            $openEastmoney = $openEastmoney -and [bool]$manualSession.open_eastmoney
        }
        if (
            $manualSession -and
            $manualSession.PSObject.Properties.Name -contains "allow_eastmoney_outside_window" -and
            [bool]$manualSession.allow_eastmoney_outside_window
        ) {
            $openEastmoney = [bool]$manualSession.open_eastmoney
        }
        $eastmoneyReady = $false
        if ($openEastmoney) {
            $eastmoneyReady = Ensure-EastmoneyTerminal
        } else {
            Stop-EastmoneyTerminalAdapter
            Stop-OwnedEastmoneyTerminal
            Stop-OwnedEastmoneyClient
        }
        if ($scheduledActive) {
            Update-TradingCalendarIfNeeded -Now $now
        }
        Ensure-QuantPaper -ManualSession $manualSession
        if ($openEastmoney -and $eastmoneyReady) {
            [void](Ensure-EastmoneyTerminalAdapter)
        } else {
            Stop-EastmoneyTerminalAdapter
        }
    } else {
        Stop-QuantPaperProcesses
        Stop-EastmoneyTerminalAdapter
        Stop-OwnedEastmoneyTerminal
        Stop-OwnedEastmoneyClient
    }

    $dateKey = $now.ToString("yyyy-MM-dd")
    $mode = if ($manualActive -and -not $scheduledActive) { "MANUAL_SESSION_OUTSIDE_WINDOW" } elseif ($manualActive) { "MANUAL_SESSION" } elseif ($scheduledActive) { "ACTIVE" } elseif ($tradingDate) { "OUTSIDE_WINDOW" } else { "NON_TRADING_DAY" }
    Write-SupervisorLog "Schedule check: $dateKey $($now.ToString('HH:mm')) mode=$mode."
}

Write-SupervisorLog "QuantPaper supervisor started. Active window: trading days 09:10-15:30 Asia/Shanghai."

do {
    Invoke-SupervisorCheck
    if ($Once) {
        break
    }
    Start-Sleep -Seconds ([Math]::Max(15, $PollSeconds))
} while ($true)
