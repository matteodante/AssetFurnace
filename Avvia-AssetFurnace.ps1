$ErrorActionPreference = 'Stop'
$url = 'http://localhost:8777/viewer/studio.html'
$ready = $false
try {
    $status = Invoke-RestMethod 'http://localhost:8777/api/catalog' -TimeoutSec 3
    $ready = $null -ne $status.schema_version
} catch {}
if (-not $ready) {
    $logs = Join-Path $PSScriptRoot 'journal'
    New-Item -ItemType Directory -Force -Path $logs | Out-Null
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) { throw 'Esegui prima install.ps1, lo script originale di AssetFurnace.' }
    Start-Process -FilePath $python -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -ArgumentList 'viewer/serve.py --host 127.0.0.1' -RedirectStandardOutput (Join-Path $logs 'server.out.log') -RedirectStandardError (Join-Path $logs 'server.err.log')
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 2
        try {
            $status = Invoke-RestMethod 'http://localhost:8777/api/catalog' -TimeoutSec 3
            if ($null -ne $status.schema_version) { $ready = $true; break }
        } catch {}
    }
}
if (-not $ready) { throw 'AssetFurnace non risponde. Controlla journal\server.err.log.' }
Start-Process $url
