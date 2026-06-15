# Publicar text2graph-evolve como repo público

Este repo **no debe publicarse con `git push` directo**. Dos razones:

1. **Corpus IMFD en los datos.** Las salidas en `results/` (trackeadas) contienen
   `evidence_quote` = substrings literales de artículos del corpus IMFD. Publicarlas
   redistribuye texto con licencia restringida.
2. **Corpus IMFD en el historial.** Aunque borres esos archivos hoy, quedan en commits
   pasados — un `git push` los expone igual.

La solución es un **export con slate limpio**: un repo nuevo, con historia fresca, que
contiene solo el código (MIT) y la documentación, sin corpus ni datos de corridas.

## Opción A — script automático (recomendado)

```powershell
# desde la raíz del repo
pwsh scripts/build_public_export.ps1
# o con destino propio:
pwsh scripts/build_public_export.ps1 -Target ..\text2graph-evolve-public
```

El script:
- copia solo `swarm_optimizer/`, `scripts/`, `docs/` y los archivos de raíz
  (`README.md`, `LICENSE`, `requirements.txt`, `CLAUDE.md`, `.gitignore`);
- **excluye** `results/`, `gold_standard*/`, `sandbox/`, `.git/`, `.env`, caches y `*.parquet`;
- **escanea** el export por `evidence_quote` y avisa si quedó texto del corpus;
- crea un repo nuevo con un único commit, listo para publicar.

Luego, para publicar:

```powershell
# 1. Crea un repo VACÍO en GitHub (sin README ni LICENSE).
# 2. Empuja el export:
cd ..\text2graph-evolve-public
git remote add origin https://github.com/<usuario>/text2graph-evolve.git
git push -u origin HEAD
```

## Opción B — manual

1. Copia a un directorio nuevo `swarm_optimizer/`, `scripts/`, `docs/`, `README.md`,
   `LICENSE`, `requirements.txt`, `CLAUDE.md`, `.gitignore`.
2. **No copies** `results/`, `gold_standard_v5/`, `sandbox/`, `.env`, ni `.git/`.
3. Verifica que no quede texto del corpus:
   ```powershell
   Get-ChildItem -Recurse -File | Select-String "evidence_quote" -List
   # no debe devolver nada
   ```
4. `git init`, `git add -A`, `git commit`, crea el repo en GitHub y `git push`.

## Checklist antes de publicar

- [ ] El export no contiene `results/` ni `gold_standard*/`.
- [ ] El escaneo de `evidence_quote` no devuelve nada.
- [ ] `LICENSE` (MIT) presente; el README aclara que el corpus IMFD no se incluye.
- [ ] No hay `.env` ni claves de API en ningún archivo.
- [ ] `docs/` no contiene quotes largas de artículos reales (revisa specs de gold sintético).
- [ ] `python -m pytest swarm_optimizer/tests/ -q` pasa en el export limpio.
