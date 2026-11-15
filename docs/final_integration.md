# Integración final del código de tesis

Estado documental: 14 de noviembre de 2026. Esta rama es una candidata local de
integración; no modifica `main`, publicaciones, datasets, checkpoints ni freezes.
La propuesta académica de referencia es `ppg_j.solorzanod_202620.pdf`, confirmada
por el investigador como la versión definitiva que tiene JP.

## Una referencia de código, fuentes científicas separadas

- Backend y CLI: `src/flir_pipeline/`. Se conservan los exploradores Streamlit de
  `apps/cluster_split_explorer.py` y `apps/detector_similarity_explorer.py`.
- Frontend: `apps/explorer/`, recuperado del subtree ya existente, con el historial
  original y los cambios locales posteriores. Se evaluó `apps/web-explorer/`;
  mantener el prefijo existente evita renombrados y referencias duplicadas.
- Laura y Manuela permanecen como productores independientes. Sus repositorios,
  modelos y resultados no se copian dentro del código central. La interfaz
  conserva autoría, commit, identidad padre/hijo, geometría, clases y asignaciones.
- Datos y resultados reales permanecen externos. `FLIR_DATA_ROOT` configura los
  originales; cada CLI recibe explícitamente sus stores, publicaciones y outputs.
  No se descubre un dataset por el nombre de la carpeta ni se reescriben recibos.

Las ocho familias Git y los cambios locales originales tienen respaldo privado
por separado. El informe operativo externo conserva inventario, SHA, patches,
bundles y pruebas de recuperación. No se versionan esos inventarios privados.

## Qué se recuperó y qué no se reaplicó

Se integraron las líneas locales de ensamblaje/NO-HUD/exploradores, cierre del
pipeline general y ejecutor automático YOLO11m. El subtree incluye la historia
del frontend publicado `fix/m02-real-data-acceptance` y los commits locales
posteriores. Los archivos sueltos de timeline y evaluación de evidencia se
recuperaron desde respaldo, no se reimplementaron.

La integración del asignador conserva las dos APIs existentes: dos destinos
para desarrollo con test fijo y tres destinos para los experimentos generales;
también conserva columnas/pesos explícitos para catálogos generales. No se
cambiaron objetivos, ratios, semillas ni asignaciones científicas existentes.
Las pruebas sintéticas de ambas familias verifican su compatibilidad.

Los commits de exportación de organización e informe detector que ya tienen
patches equivalentes en `main` no se reaplican. El ejecutor común inicial es
antecesor de implementaciones posteriores y tampoco se duplica. La rama
publicada de admisión firmada inicial es un contrato histórico distinto del
ejecutor automático posterior; no se mezclan sus recibos ni sus formatos bajo
una misma función. Su historia permanece recuperable en refs/bundles. Se
conserva el ejecutor científico firmado ya existente y sus pruebas, separado
del contrato técnico automático. Una comprobación técnica no simula una
decisión humana ni autoriza por sí sola un experimento nuevo.

## Evidencia científica que permanece distinguida

1. **Ruta de embeddings/clustering:** 1.657 ocurrencias, 1.459 contenidos;
   DINOv2 y CLIP; 36 reducciones y 414 clusterings; 60 splits cluster-aware.
   Los artefactos originales no se trasladan ni recalculan. C10 es la referencia
   documentada DINOv2/PaCMAP/DBSCAN, pendiente de ratificación/admisión final
   compatible con las restricciones actuales del test.
2. **B YOLO11m:** 75 identidades nativas a 50 épocas; 90 alias. Historical S2/S3
   son los mismos 15 modelos. S2 usa contenidos exactos balanceados y S3 relaciones
   candidatas como sensibilidad: no son clústeres CLIP/DINOv2 ni secuencias
   verificadas. El freeze y el código original de ejecución siguen siendo sus
   referencias; integrar código no reidentifica esos experimentos ni autoriza
   reanudarlos con otra firma de fuente.
3. **Contrato detector histórico:** el contrato público `detection-export-v1`
   declara 48 corridas, 16 splits y comparación histórica/random/C10/C12, con
   publicación posterior al recibo local antiguo que aún decía cero corridas
   completas. Su integridad exportada y compatibilidad de presentación pueden
   verificarse sin repetir ciencia. Esa declaración no sustituye recuperar y
   comprobar recibos nativos/configuración si se desea recertificar la ejecución.
   No es B ni se atribuye automáticamente a YOLO11m.
4. **Procedencia y variantes:** los resultados canónicos negativos se conservan;
   la procedencia de la población propia de Laura no se transfiere por nombre
   al canónico. El experimento pareado Laura sigue separado y en NO-GO.

El test protegido y la evaluación común de B pertenecen al proceso responsable;
esta integración no consulta sus nuevas salidas ni las utiliza para seleccionar
configuraciones. Las comparaciones residuales retrospectivas de clústeres y la
validation B no se presentan como una evaluación común nueva ni como efecto causal.

## Decisiones académicas vigentes

SIFT/RANSAC queda fuera del camino crítico por la decisión de JP comunicada por
el investigador. Se conserva código, visualizaciones y limitaciones como trabajo
exploratorio; no se amplía la búsqueda de fronteras o dependencias ni se usa la
geometría como restricción nueva de splitting. La temporalidad no recuperada se
declara como limitación, sin inventar PTS o secuencias.

El Comité de Ética está aprobado según confirmación del investigador para la
solicitud corregida. Solo quedan archivar/verificar constancia y condiciones de
la versión aprobada. Esa confirmación no cambia el estado de admisión del test,
los entrenamientos complementarios ni los permisos de publicación de datos.

## Uso desde un checkout nuevo

```powershell
uv sync --locked --extra dev --extra reduction --extra alignment --extra explorer
uv run flir-pipeline --help
uv run ruff check .
uv run pytest
Set-Location apps/explorer
npm ci
npm run typecheck
npm test
npm run build
npm run test:e2e
```

Estos comandos no materializan FAC ni descargan modelos. El entorno detector
CUDA es separado y se reproduce desde su versión congelada, no instalando GPU
en core. Para E2E, el servicio de auditoría usa SQLite sintético temporal y el
backend del monorepo; `FLIR_AUDIT_BACKEND` y `FLIR_AUDIT_PYTHON` permiten una
ubicación/intérprete explícitos. Los exports reales deben montarse/seleccionarse
de manera explícita; no son necesarios para la suite sintética normal.

Los resultados de las comprobaciones locales, excepciones preexistentes y las
comprobaciones todavía pendientes se entregan en el expediente de integración.
No interpretar este documento como CI verde o ejecución de GPU, Docker o test
protegido. No se incluyen dependencias, snapshots runtime privados o pesos en
la rama. La excepción pública del contrato detector histórico conserva su
allowlist y sus bytes; no se amplía a otras exportaciones reales.

## Compatibilidad y límites de la verificación local

El productor de organización conserva el checksum de pertenencia comprobada
tanto al disponer del artefacto completo como al recuperar membresías congeladas.
El consumidor reconoce la procedencia opcional sin convertir metadata recuperada
en un artefacto completo. La paridad se prueba con exports del backend sintético,
no copiando un runtime privado. El schema público refleja el modelo vigente;
las exportaciones científicas anteriores no se regeneran.

Los contratos públicos del detector mantienen sus bytes LF mediante atributos
Git: la conversión CRLF de Windows impediría verificar sus checksums aunque el
JSON representara los mismos valores. No se modifica el contenido publicado.

La suite E2E local conserva fallos de accesibilidad y posicionamiento reproducidos
en una copia del frontend original. El recorrido de candidatos usa un escenario
sintético separado, con referencias y recibos validados por el contrato completo;
no altera la salida del productor utilizada para las pruebas de paridad ni crea
evidencia real. Prettier global también falla en el baseline. Son límites
explícitos de la candidata; no se deshabilitan tests,
se alteran expectativas ni se declara CI verde. Las pruebas opt-in que necesitan
publicaciones reales requieren sus montajes y autorización propios.

## Fusión y reversión

La revisión debe comparar esta rama con el `main` remoto registrado y comprobar
tests, trazabilidad y privacidad. Solo con autorización posterior se hará push
o merge a `main`. Los commits de integración son locales. No hay cambios a
revertir en los repositorios originales: conservar el checkout candidato y sus
commits permite revisar o abandonar la propuesta sin resetear, limpiar ni mover
los originales. Las copias de recuperación se verificaron en repositorios
independientes; no se ejecuta una restauración sobre trabajo activo.
