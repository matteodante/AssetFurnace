param([string]$Root = $PSScriptRoot, [int]$Port = 8777)
# Ferma il server avviato da Avvia-AssetFurnace. Con lui si chiudono sd-cli, llama-server e
# gli altri lavori che ha lanciato (viewer/serve.py li tiene in un job object di Windows).
$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $Root "journal\server-$Port.pid"
if (-not (Test-Path -LiteralPath $pidFile)) {
    Write-Host "AssetFurnace non risulta avviato sulla porta $Port."
    exit 0
}
$serverPid = [int]((Get-Content -LiteralPath $pidFile -Raw).Trim() -split '\s+')[0]
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $serverPid" -ErrorAction SilentlyContinue
if ($null -eq $process -or $process.CommandLine -notmatch 'serve\.py') {
    # il pid non e' piu' il server: Windows riusa i numeri, quindi non si ferma nulla
    Write-Host "Il server sulla porta $Port non e' in esecuzione."
    Remove-Item -LiteralPath $pidFile -Force
    exit 0
}
Stop-Process -Id $serverPid -Force
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "AssetFurnace fermato."
