[CmdletBinding()]
param(
    [switch]$KeepEastmoney
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$runtimePath = Join-Path $projectRoot "data\runtime"
$manualSessionPath = Join-Path $runtimePath "manual-session.json"
$supervisorPath = Join-Path $PSScriptRoot "quantpaper_supervisor.ps1"
$powershellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

Remove-Item -LiteralPath $manualSessionPath -Force -ErrorAction SilentlyContinue
$normalizedPython = [IO.Path]::GetFullPath($pythonPath)
foreach ($process in @(
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ExecutablePath -and
            [String]::Equals($_.ExecutablePath, $normalizedPython, [StringComparison]::OrdinalIgnoreCase) -and
            $_.CommandLine -match "quantpaper\.cli\s+(serve|runner)"
        }
)) {
    Stop-Process -Id ([int]$process.ProcessId) -Force -ErrorAction SilentlyContinue
}
Get-ChildItem -LiteralPath $runtimePath -Filter "quantpaper-*.pid" -File -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

if (-not $KeepEastmoney) {
    & $powershellPath -NoProfile -STA -ExecutionPolicy Bypass -WindowStyle Hidden -File $supervisorPath -Once
}
Write-Output "QuantPaper paper process closed."
