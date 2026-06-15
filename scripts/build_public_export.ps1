<#
.SYNOPSIS
  Construye un export PÚBLICO de text2graph-evolve con slate limpio: historia de git
  fresca y SIN corpus IMFD ni datos de corridas.

.DESCRIPTION
  Copia solo la superficie de producción (swarm_optimizer/, scripts/, docs/ y los
  archivos de raíz) a un directorio hermano, NUNCA results/, gold_standard*/, sandbox/
  ni .git/. Escanea el resultado por `evidence_quote` (texto del corpus) como guardia
  anti-fuga, y crea un repo nuevo con un único commit listo para publicar.

  El historial actual y los results/ trackeados embeben evidence_quote (substrings
  literales de artículos IMFD); por eso NO se debe publicar este repo con `git push`
  directo, sino este export limpio.

.EXAMPLE
  pwsh scripts/build_public_export.ps1
  pwsh scripts/build_public_export.ps1 -Target ..\mi-repo-publico
#>
param(
    [string]$Target = "..\text2graph-evolve-public"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path "swarm_optimizer")) {
    Write-Error "Corre este script desde la raíz del repo (donde está swarm_optimizer/)."
    exit 1
}
if (Test-Path $Target) {
    Write-Error "El destino '$Target' ya existe. Bórralo o usa -Target con otra ruta."
    exit 1
}

New-Item -ItemType Directory -Path $Target | Out-Null
$Target = (Resolve-Path $Target).Path
Write-Host "Export limpio -> $Target"

# Superficie de producción. NUNCA: results/, gold_standard*/, sandbox/, .git/, .env
$dirs  = @("swarm_optimizer", "scripts", "docs")
$files = @("README.md", "LICENSE", "requirements.txt", "CLAUDE.md", ".gitignore")

foreach ($d in $dirs)  { if (Test-Path $d) { Copy-Item $d -Destination $Target -Recurse } }
foreach ($f in $files) { if (Test-Path $f) { Copy-Item $f -Destination $Target } }

# Barrido de caches y cualquier dato que se haya colado.
Get-ChildItem $Target -Recurse -Directory -Force |
    Where-Object { $_.Name -in @("__pycache__", ".pytest_cache", "results", "sandbox") } |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $Target -Recurse -File -Force |
    Where-Object { $_.Extension -in @(".pyc", ".parquet") } |
    Remove-Item -Force -ErrorAction SilentlyContinue

# Guardia anti-fuga: ningún archivo del export debe contener texto del corpus.
$leak = Get-ChildItem $Target -Recurse -File |
    Select-String -Pattern "evidence_quote" -List -ErrorAction SilentlyContinue
if ($leak) {
    Write-Warning "Posible texto de corpus (evidence_quote) en el export:"
    $leak | ForEach-Object { Write-Warning "  $($_.Path)" }
    Write-Warning "Revisa/elimina estos archivos ANTES de publicar."
} else {
    Write-Host "Guardia anti-fuga: OK (sin evidence_quote en el export)."
}

# Historia fresca: un solo commit, sin rastro del corpus en el pasado.
Push-Location $Target
git init -q
git add -A
git commit -q -m "Initial public release: text2graph-evolve (codigo MIT, sin corpus)"
Pop-Location

Write-Host ""
Write-Host "Listo. Para publicar:"
Write-Host "  1. Crea un repo VACIO en GitHub (sin README ni LICENSE)."
Write-Host "  2. cd `"$Target`""
Write-Host "  3. git remote add origin https://github.com/<usuario>/text2graph-evolve.git"
Write-Host "  4. git push -u origin HEAD"
