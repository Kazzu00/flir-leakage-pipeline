# FLIR Audit & Human Review v1

Módulo local de **auditoría exploratoria y juicio humano**, integrado en
`flir-pipeline-explorer` → Sequences & linkage → Human audit review.
No crea secuencias, grupos de independencia, particiones ni entrenamiento.

## Inspección y decisiones de arquitectura

Puntos de partida: backend `8a8100ff6ea1796b2eb694f0ad317e57b01431c8`, rama
`feat/hypatia-organization-evidence-export`; frontend
`74d29fdbf6c9b67f27b679f28608a4bb2772b49c`, rama `fix/m02-real-data-acceptance`.
En ambos se creó `feat/audit-human-review-v1`. Se preservaron los cambios previos
en `explorer/organization_contract.py` y `docs/analysis/repository_audit.md`.
El frontend estaba limpio. Las fuentes vigentes son `docs/status.md` y
`docs/pipeline_traceability.md`, no los nombres anteriores del contexto permanente.

Se inspeccionaron CLI, dependencias, pruebas, documentación, los contratos de
organización, revisión de experimentos y la ruta React `SequenceExplorer`.
Los revisores existentes consumen otros contratos científicos; no se convierten
estas alertas legacy en sus secuencias o enlaces confirmados.

Se reutilizan Typer, Pydantic, Pillow, React, React Query y Zod. Sin dependencias
nuevas. SQLite y el servidor HTTP pertenecen a la biblioteca estándar de Python.
El adaptador frontend valida el contrato; los componentes no interpretan los
JSON/CSV originales. Esta API local de anotaciones es independiente de los
snapshots científicos de solo lectura y no modifica sus contratos.

## Semántica

- **Observación:** índice histórico o posición nominal reportada por un registro.
  Las 780 filas del perfil son transiciones, no una nueva extracción de 781 imágenes.
- **Alerta:** evento candidato emitido por un método exploratorio; no es verdad
  de referencia ni una decisión humana. Las clasificaciones originales se conservan
  en los bytes fuente, sin promoverlas al historial humano.
- **Continuidad audiovisual:** `continuous`, `shot_change`, `uncertain`.
- **Dependencia visual/región:** `same_region`, `different_region`, `uncertain`.
  `shot_change + same_region` es válido. Un corte no prueba independencia: puede
  cambiar la toma sin cambiar la región observada.
- Un vídeo puede contener varias tomas, recurrencias y movimientos; no constituye
  automáticamente una unidad indivisible para un futuro split.
- Los tiempos son reportados/nominales. Precisión no declarada = `null`.
  No se aplica automáticamente la hipótesis `1978 + índice`, ni se inventan
  timestamps o atribuciones de vídeo a partir del nombre del archivo.
- `zoom`, `camera_motion`, `gain_change`, `polarity_change`, `signal_loss` y
  `unknown` describen fenómenos observados; no determinan independencia.

El contrato `audit-evidence-v1` está definido en `audit/models.py` y en el
[JSON Schema](../contracts/audit-evidence-v1.schema.json). Cada evento contiene
archivo, localizador de registro, método, versión declarada o `unreported`,
observaciones y referencias de imagen. La versión del importador se distingue
de la versión del método original. No se crean `frame_id` ni `content_id`
ficticios cuando las fuentes solo proporcionan índices históricos.

## Instalación y smoke sin FAC (PowerShell)

Backend, Python 3.11 y uv:

```powershell
uv sync --extra dev
uv run flir-pipeline sequences audit --help
uv run flir-pipeline sequences audit demo reports/audit-demo
uv run flir-pipeline sequences audit summary --db reports/audit-demo/audit.sqlite3
uv run flir-pipeline sequences audit verify --db reports/audit-demo/audit.sqlite3 --originals
uv run flir-pipeline sequences audit serve --db reports/audit-demo/audit.sqlite3
```

El directorio de demo debe ser nuevo. Contiene tres láminas **SYNTHETIC**, sin
vídeo real, y tres eventos sin decisiones. No simula resultados FAC.
En otra terminal, desde el frontend:

```powershell
npm ci
npm run dev -- --port 5173 --strictPort
```

Abra `http://127.0.0.1:5173/organization/sequences?mode=audit`.
Seleccione un candidato, inspeccione la lámina y marque explícitamente qué
evidencia consultó. Registre revisor, justificación, dimensiones, confianza y
estado. Guarde; la versión aparecerá en el historial. Modificar y guardar crea
otra versión. Cierre navegador y servicio, reabra ambos y compruebe el historial.
La persistencia está en SQLite, no en `localStorage`. Una revisión puede quedar
pendiente y expresar incertidumbre. Solo se registran las imágenes que el usuario
marca como consultadas; no se asume que cargar una imagen equivale a revisarla.

## Artefactos existentes

Use un directorio de auditoría externo, sin escribir en los datos originales:

```powershell
$auditRoot = Read-Host 'Directorio local con los artefactos exploratorios'
uv run flir-pipeline sequences audit validate $auditRoot --priority-only
uv run flir-pipeline sequences audit import $auditRoot --db reports/audit-local/audit.sqlite3 --priority-only
uv run flir-pipeline sequences audit summary --db reports/audit-local/audit.sqlite3
uv run flir-pipeline sequences audit verify --db reports/audit-local/audit.sqlite3 --originals
uv run flir-pipeline sequences audit serve --db reports/audit-local/audit.sqlite3
```

`--priority-only` selecciona los nombres prioritarios del encargo y reporta
los demás como excluidos. Sin esa opción se intentan todos los JSON/CSV/ZIP
del directorio y los formatos desconocidos se reportan como errores.
Ausencias se reportan sin fabricar filas; errores de contenido producen salida 1.
Cada archivo válido se importa en una transacción independiente; un archivo
erróneo no aporta filas parciales ni impide conservar otros archivos válidos.
El informe se guarda en la base de datos. Repetir la importación no duplica filas.

Adaptadores observados: inventario de índices por split, perfil temporal,
candidatos 5 FPS, rachas y manifiestos JSON/CSV de ventanas dentro de ZIP.
Se validan campos, números finitos, rangos, intervalos, conteos declarados,
booleanos y concordancia JSON/CSV. JSON tiene precedencia para normalizar;
el CSV se conserva como evidencia alternativa sin duplicar eventos.
Las ventanas extendidas sin vídeo declarado conservan `source_video=null`.
La clasificación automática y las métricas no cambian con una decisión humana.

## Persistencia, exportación y seguridad

SQLite conserva bytes originales de cada artefacto aceptado, SHA-256, ruta local,
representación normalizada y revisiones inmutables. IDs de registros dependen
del archivo, bytes y localizador: cambios de evidencia generan IDs nuevos y
no heredan decisiones. Ventanas de distintos paquetes permanecen separadas;
coincidir en un tiempo nominal no demuestra identidad de evento.
La importación no confirma ninguna alineación ni dependencia.

Guardado transaccional y `expected_version` evitan perder revisiones en cierres
o escrituras concurrentes. Exportación mediante archivo temporal + reemplazo
atómico; se rechazan destinos ya existentes. No guardar exportaciones dentro
de los directorios originales.

```powershell
uv run flir-pipeline sequences audit export reports/audit-local/decisions-v1.json --db reports/audit-local/audit.sqlite3
uv run flir-pipeline sequences audit export reports/audit-local/decisions-v1.csv --db reports/audit-local/audit.sqlite3 --format csv
uv run flir-pipeline sequences audit import-decisions reports/audit-local/decisions-v1.json --db reports/audit-local/audit.sqlite3
```

JSON y CSV preservan **todas** las revisiones, no solo la última. CSV incluye
`decision_json` para conservar listas y tipos exactamente. Reimportar valida
evento, SHA de evidencia y cadena de versiones; rechaza conflictos o referencias
ausentes. Importe primero los mismos artefactos si usa una base nueva.
La revisión no acredita la identidad real del revisor: es un identificador
declarado por el usuario local, no una firma digital.

El servicio escucha exclusivamente en `127.0.0.1`, puerto 8765. Solo permite
el origen frontend local declarado (por defecto `http://127.0.0.1:5173`), valida
Host, exige JSON y cabecera de revisión para escrituras y sirve imágenes por
referencia de evento, nunca por una ruta arbitraria. No está diseñado como
servicio multiusuario ni para exposición en red. El frontend usa el puerto
8765; `--port` permite diagnóstico manual de la API en otro puerto.

No se extraen ZIP al sistema de archivos, no se leen vídeos completos y no se
cargan todas las imágenes decodificadas. Límite por artefacto: 64 MiB; por miembro
ZIP: 16 MiB; se rechazan rutas inseguras, enlaces y nombres duplicados. Una lámina
se lee bajo demanda del snapshot del ZIP. Los originales pueden desconectarse
sin perder la evidencia ya importada. `verify --originals` comprueba además
que las rutas fuente sigan presentes y tengan los mismos hashes.
DB, imágenes, exports y reportes deben permanecer locales e ignorados por Git.

## Verificación y límites

```powershell
uv run ruff check .
uv run python -m pytest tests/test_audit_review.py -q
```

En el frontend, con el backend instalado en el checkout hermano (o configure
`FLIR_AUDIT_BACKEND` con su ruta local):

```powershell
npm run lint
npm run typecheck
npm run test
npm run build
npm run test:e2e -- e2e/audit-review.spec.ts
```

La prueba E2E inicia un servicio y una base sintéticos, muestra una imagen real
del fixture, guarda/corrige decisiones, reinicia el proceso y compara exports.
Usa loopback sin mocks HTTP. El suite pytest conserva su bloqueo de red;
la integración de navegador se ejecuta por separado. Libere el puerto 8765
antes de ejecutar esta prueba.

No hay reproducción de vídeo ni extracción adicional: el revisor muestra
las láminas antes/después y de contexto realmente disponibles. Los eventos
sin lámina muestran ausencia explícita; no se vinculan imágenes por proximidad.
Los JSON gap3/clasificaciones/findings no estaban disponibles y sus adaptadores
permanecen pendientes hasta inspeccionar sus esquemas. Tampoco se admiten aún
los cinco auxiliares de alineación y métricas de fronteras detectados.
No se hizo revisión humana real ni se verificó sistemáticamente la alineación
de validation/test. La siguiente fase debe resolver esas ausencias y revisar
las ventanas con protocolo y revisor identificados, antes de diseñar un grafo
de dependencias. No ajustar umbrales con el test final del detector.
