# Fase 2B: revisión de pares temporales y visuales

La unidad es un par de ocurrencias históricas, no una secuencia ni un componente
del futuro grafo. `correct_match` valida procedencia imagen-vídeo; nunca se convierte
en `continuous` ni `same_region`. Ambos ejes se anotan independientemente. Una toma
puede cambiar conservando región. Ausencia de SIFT no demuestra región diferente.

## Preparación reproducible

Usar CPU y el entorno de visión ya instalado. `FLIR_DATA_ROOT` debe apuntar a los
originales FAC; no se leen etiquetas YOLO. Definir `ALIGNMENT_REVIEWS` y
`ALIGNMENT_CLOSURE` en la sesión con los archivos recibidos. El comando exige un
directorio nuevo y verifica el vídeo, el piloto fuente, la cadena de versiones y
los hashes individuales de las evidencias declaradas en el export humano.

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences alignment dependencies prepare reports/phase2a1-pilot $env:ALIGNMENT_REVIEWS $env:ALIGNMENT_CLOSURE $env:FLIR_DATA_ROOT reports/phase2b-reproduction --seed 20261103
```

El conjunto entregado está en `reports/phase2b-final`, acompañado por
`reports/phase2b-selection.md` y `reports/phase2b-report.json`. Son artefactos
locales privados e ignorados por Git. `pilot.json` y `checksums.json` fijan selección,
métricas, medios y referencias congeladas. El informe de cierre se conserva como
declaración retrospectiva del revisor: no amplía `evidence_consulted` original.
`reports/phase2b-pilot` corresponde a una ejecución diagnóstica anterior con máscara
demasiado restrictiva; no es el conjunto para anotar.

Selección dirigida sobre las 24 procedencias aceptadas por una persona: cuatro
fronteras; seis cercanos con separación nominal ≤60 s, repartidos sobre el rango de
puntos medios disponible; seis no consecutivos (≥10 s), priorizando soporte
geométrico/cross-split y correlación central; cuatro cambios candidatos, ordenados
por diferencia de media y contraste dentro de 60 s; cuatro controles con ocho
imágenes diferentes, poco soporte geométrico y apariencia fotométrica parecida.
El orden de categorías evita repetición de pares. La semilla resuelve empates.
No se afirma cobertura de todo el vídeo ni se sustituyen déficits silenciosamente.

Se calculan 276 comparaciones SIFT/RANSAC como máximo para las 24 imágenes.
La máscara conserva el rectángulo central de 70% × 60%, excluye verde dominante
`G−R>25, G−B>20` dilatado 7 px y exige distancia al borde de máscara de
`max(12, 3*tamaño_keypoint)` px. Se muestran las máscaras. Texto gris y residuos
pueden persistir; el revisor debe distinguir terreno y HUD. No se optimizó con
etiquetas de test. Una homografía requiere ≥8 inliers y comprobaciones geométricas
para mostrar superposición, sin que eso confirme identidad. Las métricas de
alineamiento anteriores y las nuevas métricas entre imágenes son distintas.

## Sesiones independientes

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences alignment dependencies serve reports/phase2b-final reports/phase2b-reviewer1.sqlite3 --role reviewer1 --port 8767
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences alignment dependencies serve reports/phase2b-final reports/phase2b-reviewer2.sqlite3 --role reviewer2 --port 8768
```

Ejecutar cada servicio en una terminal diferente. Abrir `http://127.0.0.1:8767` y
`http://127.0.0.1:8768`. Solo se escucha en loopback y se comprueba origen/Host.
El segundo servicio accede a otro ledger y únicamente a los ocho pares asignados:
las cuatro fronteras y cuatro cambios difíciles. Su API no entrega respuestas del
primer revisor. No es una barrera contra alguien con acceso completo al portátil:
la independencia requiere también no compartir exportaciones ni abrir la otra sesión.

1. Examinar originales, top-1 de procedencia, contexto anterior/posterior, máscaras
   y correspondencias. Los splits están inicialmente ocultos.
2. Abrir los metadatos para tiempos, incertidumbre, hashes y revisión de alineación.
   Los ±1 s son contexto local, no prueba de continuidad durante todo el intervalo.
3. Confirmar individualmente los materiales consultados, o pulsar explícitamente
   «Confirmo haber examinado todos los materiales mostrados». Cargar imágenes no
   activa casillas. Casillas previamente guardadas sí se recuperan.
4. Registrar ambos ejes, confianza, fenómenos, justificación y revisor. Puede dejarse
   ambos como `uncertain` con confianza indeterminada. Un estado pendiente admite
   guardar trabajo parcial. Una revisión requiere originales A/B consultados y
   justificación; una conclusión requiere confianza explícita.
5. Guardar versión. Una corrección añade historial; una versión concurrente obsoleta
   se rechaza. «Siguiente pendiente» navega por la asignación.

Las anotaciones previas no se modifican. Descargar JSON o CSV desde la interfaz.
CSV conserva cada decisión completa en `decision_json`, con Unicode y hashes;
no es una tabla que pierda revisiones anteriores. Los contratos públicos están en
`docs/contracts/dependency-*.schema.json`; regenerarlos con
`uv run --no-sync python scripts/export_dependency_schemas.py`.

## Comparación y adjudicación

```powershell
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences alignment dependencies report reports/phase2b-final reports/phase2b-reviewer1.sqlite3 --reviewer2-db reports/phase2b-reviewer2.sqlite3
uv run --no-sync python -c "from flir_pipeline.cli import app; app()" sequences alignment dependencies serve reports/phase2b-final reports/phase2b-adjudication.sqlite3 --role adjudicator --reviewer1-db reports/phase2b-reviewer1.sqlite3 --reviewer2-db reports/phase2b-reviewer2.sqlite3 --port 8769
```

Abrir adjudicación únicamente después de las dos revisiones independientes. Se
seleccionan explícitamente las dos decisiones que se adjudican; se conserva una
tercera historia, sin sobrescribirlas. Reiniciar esa sesión si aparecen nuevas
revisiones originales: sus referencias se congelan al abrirla. El reporte puede
añadir `--adjudication-db` para incluir la decisión de adjudicación.

Concordancia es `null` mientras no existan pares efectivamente revisados por ambos.
Después informa acuerdos y desacuerdos observados por eje, con denominador y IDs,
sin afirmar calibración ni tasas poblacionales. No derivar leakage, independencia
de secuencias o efecto sobre un detector únicamente de `same_region`.

## Verificación

```powershell
uv run --no-sync python -m ruff check .
uv run --no-sync python -m pytest tests/test_dependency_pilot.py tests/test_alignment_pilot.py tests/test_alignment.py tests/test_audit_review.py tests/test_audit_v11.py tests/test_smoke.py -q
node scripts/check_dependency_pilot_browser.cjs
uv build --wheel --out-dir reports/phase2b-wheel
```

El chequeo de navegador utiliza Playwright instalado en el frontend local hermano.
Escribe decisiones solamente en fixtures sintéticos; exige cero decisiones reales
y verifica persistencia tras reinicio. **No ejecutarlo después de iniciar la
anotación real**, porque su precondición de cero decisiones ya no se cumpliría.
Los tests pytest son sintéticos, offline y sin GPU. Esta fase no cambia frontend
React/TypeScript: reutiliza el servidor local de Human Alignment y añade una vista
HTML empaquetada con Python.
