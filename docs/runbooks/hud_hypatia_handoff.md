# Entrega operativa HUD / NO-HUD — 2026-11-04

## Dictamen

**BASE está materializado y verificado técnicamente; los dos entrenamientos aún
no están habilitados.** NO-HUD tiene imágenes reales, pero cero transferencias de
anotaciones verificadas en la publicación inspeccionada. No existe todavía una
población emparejada entrenable. Ningún job fue enviado ni se accedió a Hypatia.

La rama persistente es `chore/integrate-hypatia-base`, en el worktree hermano
`flir-leakage-integration`. Pertenece al mismo repositorio Git canónico. El checkout
original conserva su rama y sus dos cambios locales preexistentes. No hubo push,
merge a main, eliminación de ramas, modificación de FAC ni apertura de payloads
protegidos. Las fuentes operativas son `docs/status.md` y
`docs/pipeline_traceability.md`; no existen los nombres antiguos del contexto.

## Datos y cobertura observados

Se reutilizaron `FLIR-DEV-CURADO-v1`, la publicación fixed-test de cinco semillas,
la preparación `curated_development_v1` y el ejecutor `common-executor`.
No se recalcularon splits, clustering ni auditorías exploratorias.

| Propiedad | BASE preparado | NO-HUD |
| --- | ---: | ---: |
| Contenidos originales de desarrollo | 1.135 | Sin población anotada admitida |
| Imágenes en inventario NO-HUD previo | No aplica | 19.800 |
| Imágenes/labels incluidos en esta entrega | 1.135 / 1.135 | 0 / 0 |
| Pares HUD/no-HUD con cajas verificadas | 0 | 0 |
| S1 train / val conservado | 1.044 / 91 | No definido |
| S2 train / val conservado | 1.065 / 70 | No definido |
| S3 train / val conservado | 1.052 / 83 | No definido |
| Etiquetas vacías pendientes | 185 | No se fabricaron negativos |
| Otras etiquetas | 950 originales, sin nueva adjudicación visual | 0 transferidas |
| Solapamiento exacto con referencia protegida | 0 | No se materializó |
| Listo científicamente para entrenar | No | No |

BASE conserva las clases originales, incluido ID 4=`SDZI`; su equivalencia con
"Heavy Machinery" sigue sin adjudicar. La verificación actual decodificó cada
imagen, comprobó dimensiones, pares, SHA256, clases, cajas normalizadas, duplicados
de cajas y unicidad de contenido. Contrastó los localizadores y hashes contra los
registros originales y las tres asignaciones contra sus membresías con recibo.
La reserva comprende 514 ocurrencias / 373 contenidos; solo se leyeron metadatos.
La igualdad de hashes no certifica ausencia de correlación visual o temporal.

Procedencia de BASE: 1.087 contenidos de `Imagenes.zip` y 48 de `Adendo.zip`.
Dimensiones observadas: 578 imágenes de 1920×1080 y 557 de 1280×720. Es la
población ampliada curada existente; no debe confundirse con el histórico puro.

Los ocho conflictos conocidos están excluidos de esta población por la reserva
previa. No se adjudicaron ni cambiaron. Las 57 ocurrencias duplicadas retiradas
del desarrollo original no son nuevos contenidos. Deduplicar cambia el peso de
observaciones y debe declararse al comparar con experimentos por ocurrencias.

## NO-HUD y transferencia

Existen seis ZIP `frames_sin_hud-*`, inventario y fingerprints en la raíz FAC,
además de tres videos fuente en `fac/Videos`. No se extrajeron masivamente ni se
repitió matching. La publicación `no_hud_label_transfer_v1/review-20261015-a`
contiene 19.800 pendientes, cero verificadas, cero labels candidatos y 4.578
imágenes con candidatos protegidos. Se verificaron los hashes de sus cuatro
artefactos de metadatos principales contra el recibo, sin leer galerías protegidas.
Esto no es una nueva verificación exhaustiva de todos los ZIP.

Se leyó el export humano reciente: 27 revisiones históricas y 24 decisiones
finales `correct_match`, todas con top-1 aceptado. No son una aprobación de cajas,
geometría ni integridad de objetos.
No se convirtieron en transferencias autorizadas.

Se recuperó del stash el código existente `data/no_hud.py`, `no_hud_alignment.py`,
su CLI, pruebas y runbooks; el stash permanece intacto. La ruta A soporta identidad
o escalado uniforme, evidencia atribuida de frame nativo, comparación fuera de
regiones modificadas y rechazo de cajas afectadas. La ruta B mantiene candidatos
temporales/visuales; recorte, traslación, homografía, cambios de toma y procesamiento
con pérdida **no están habilitados para transferir cajas**. Falta una transformación
validada por pareja y revisión del resultado, especialmente para objetos móviles.
No se implementó una transformación arbitraria sin evidencia que permita validarla.

Los estados pendientes y rechazados nunca se convierten en labels de entrenamiento.
`excluded_pairs.csv` enumera los 1.135 contenidos BASE excluidos de una comparación
pareada por falta de transferencia verificada. `pending_annotations.csv` conserva
el estado individual de revisión. Los identificadores `source_content_id` y
`target_content_id` deben conservar hashes distintos si cambian los bytes;
la paridad se expresará mediante una identidad de observación compartida.

El common-test mantiene su composición protegida. Su eventual versión NO-HUD debe
tener correspondencia 1:1 y cajas verificadas por el custodio, sin intervenir en
selección de parámetros. Hoy esa versión no existe ni está aprobada.

## Paquete local y comprobación

Artefactos privados: `tesis-data/operations/hud-handoff-20261104/`, fuera de Git:

- `base_hud/`: imágenes, labels, manifest, procedencia, matriz original, tres YAML
  y listas, recibo de desarrollo. Sigue `DEVELOPMENT_NOT_ADMITTED`.
- `base_hud.tar`: 2.283 miembros verificados por hash tras empaquetar; `SHA256SUMS`.
- `no_hud/status.json`: **paquete de estado bloqueado**, no dataset entrenable.
- `validation.json`, exclusiones y pendientes; logs de pruebas y builder local.
- `experiment_proposal.json`: propuesta común no admitida, semilla detector 42,
  split seed 0, política S3 propuesta, hiperparámetros heredados del executor;
  batch y población emparejada pendientes. No es una configuración ejecutable.

Los YAML BASE apuntan al destino propuesto `/work/flir/hud-comparison/base_hud`.
Ese destino **no está confirmado en Hypatia**. No editar los YAML ni el recibo a
mano: si se elige otro destino hay que regenerar el paquete con `--linux-root` y
recalcular el TAR/checksums. La matriz existente contiene nueve runs propuestos;
no se la presenta como contrato válido de dos runs HUD/no-HUD.

Desde la rama integradora, con `$DataRoot` apuntando a la raíz externa:

```powershell
$Dataset = Join-Path $DataRoot 'datasets/FLIR-DEV-CURADO-v1'
$Prior = Join-Path $DataRoot 'operations/hypatia-yolo11m-preflight-20261027'
$Publication = Join-Path $DataRoot 'audits/fixed_test_train_val_v1/five-seeds-20261013-a'
$Handoff = Join-Path $DataRoot 'operations/hud-handoff-20261104'
# Repetir verifica idempotencia; otro destino Linux exige una salida local nueva.
uv run --offline --no-sync flir-pipeline detection common-executor prepare-curated-development `
  $Dataset "$Dataset/reports/manifest.csv" "$Prior/matrix.json" `
  "$Prior/provenance.json" "$Handoff/base_hud" `
  --linux-root '/work/flir/hud-comparison/base_hud' `
  --protected-records "$Publication/sources/proposal/protected_records.parquet" `
  --protected-receipt "$Publication/receipt.json"
$ReceiptHash = (Get-FileHash "$Handoff/base_hud/development_receipt.json" -Algorithm SHA256).Hash.ToLower()
uv run --offline --no-sync python scripts/hypatia/verify_hud_package.py `
  "$Handoff/base_hud" --receipt-sha256 $ReceiptHash
```

## Transferencia y Hypatia manual

El registro previo identifica `hypatia.uniandes.edu.co`; no acredita conectividad
actual, usuario, almacenamiento, cuotas ni GPU. No se ejecutó SSH. Definir los
parámetros con los valores reales de la cuenta. El checkpoint local existe bajo
`models/yolo11m/yolo11m.pt`; su hash quedó en la evidencia privada. **Origen oficial
y compatibilidad no verificados; no se cargó ni transfirió.**

```powershell
$SshTarget = '<USUARIO>@hypatia.uniandes.edu.co'
$RemoteDirectory = '/<DIRECTORIO_EXISTENTE_AUTORIZADO>'
# Crear antes el directorio en la sesión SSH manual.
./scripts/hypatia/transfer_hud_handoff.ps1 -PackageRoot $Handoff `
  -SshTarget $SshTarget -RemoteDirectory $RemoteDirectory
# Transferir el código consolidado sin datos ni entornos mediante el bundle local:
scp "$Handoff/flir-integration.bundle" "${SshTarget}:${RemoteDirectory}/"
```

En SSH manual, activando el entorno Python de detección ya configurado:

```bash
export TRANSFER_DIR='/<DIRECTORIO_EXISTENTE_AUTORIZADO>'
export FLIR_REPO='/<CHECKOUT_NUEVO>'
cd "$TRANSFER_DIR"
git clone --branch chore/integrate-hypatia-base flir-integration.bundle "$FLIR_REPO"
sha256sum -c <(grep '  base_hud.tar$' SHA256SUMS)
# Extraer solo en destino nuevo autorizado. No sobrescribir otra preparación.
test ! -e base_hud && tar -xf base_hud.tar
sha256sum -c SHA256SUMS
RECEIPT_SHA=$(awk '$2=="base_hud/development_receipt.json" {print $1}' SHA256SUMS)
python "$FLIR_REPO/scripts/hypatia/verify_hud_package.py" \
  "$TRANSFER_DIR/base_hud" --receipt-sha256 "$RECEIPT_SHA"
export PYTHONPATH="$FLIR_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export FLIR_PYTHON="$(command -v python)"  # Python del entorno de detección activado
export FLIR_EXECUTOR_OUTPUT='/<SALIDA_NUEVA_AUTORIZADA>'
cd "$FLIR_REPO"
# Preflight GPU en un nodo asignado; un login sin GPU no valida capacidad.
bash scripts/hypatia/common_test_yolo11m.sh preflight \
  "$FLIR_EXECUTOR_OUTPUT/preflight.json"
# Probe sintético opcional en SLURM: no utiliza FAC ni test.
export FLIR_EXECUTOR_MODE=probe
bash scripts/hypatia/common_test_yolo11m_submit.sh \
  --partition="$GPU_PARTITION" --gres=gpu:1 --cpus-per-task="$CPUS" \
  --mem="$RAM" --time="$WALLTIME"
# JOB_ID es el valor real devuelto por sbatch, nunca uno inventado.
squeue -j "$JOB_ID"
sacct -j "$JOB_ID" --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

El preflight existente registra Torch, torchvision, Ultralytics, CUDA y hardware;
exige sus versiones contractuales. No instala dependencias ni descarga modelos.
Se intentó localmente con `--cpu`: el entorno core no tiene `torch` y terminó
con `ModuleNotFoundError`, antes de producir el recibo de entorno. Se conserva
`local-cpu-preflight.log`; no se afirma preflight local aprobado ni se instalaron
dependencias pesadas para sustituir la comprobación del clúster.
`base_hud.sbatch` y `no_hud.sbatch` son **plantillas bloqueadas**, separadas por
variante. Ejecutan preflight y terminan con código 2; no entrenan. No enviarlas
esperando una corrida. El protocolo firmado existente permite nueve runs de otra
población; reutilizarlo sin adaptar la identidad experimental daría una comparación
incorrecta. No hay hoy comandos honestos que lancen las dos corridas solicitadas.

## Qué falta y qué se puede mostrar mañana

Para una corrida preliminar BASE: decidir el tratamiento de las 185 vacías y el
alcance de revisión de las otras 950, fijar población/política/semilla/presupuesto
y soportar esa población explícitamente en el ejecutor sin liberar test.
Para la comparación: validar transferencias NO-HUD, construir BASE emparejado,
congelar un único conjunto de hiperparámetros/augmentations y validar capacidad
GPU. No basta retirar un flag de bloqueo ni copiar cajas del vecino más parecido.

Se pueden mostrar la preparación reproducible, la tabla de cobertura, los checksums,
la exclusión exacta de la referencia protegida, el inventario de pendientes y la
integración del código. **No se pueden mostrar métricas YOLO11m nuevas, un efecto
HUD demostrado, ni una evaluación final del detector.**

## Consolidación y organización

La rama integradora avanzó desde `6b6f92c` a la preparación curada `6f349c5`, y
combinó los commits de auditoría/alineamiento hasta `bdc9be5`. Los conflictos eran
adiciones documentales; se conservaron ambos bloques. El frontend `e00cf94` se
incorporó con subtree sin squash, preservando historia y `src/` del backend.
Su CI se activó desde `.github/workflows/explorer.yml`; Docker conserva su contexto
en `apps/explorer`. Se corrigió una prueba que dependía de un manifest privado
ignorado, reemplazándolo por una fixture sintética. No se creó otra interfaz.

Inventario acotado del escritorio y clasificación para transición:

| Carpetas | Clasificación / acción |
| --- | --- |
| flir-leakage-pipeline, flir-leakage-integration, flir-leakage-backend-v2 | ACTIVE: mismo repositorio/worktrees; conservar |
| flir-pipeline-explorer | ACTIVE: origen histórico conservado después del subtree |
| tesis-data | ACTIVE: datos, cachés, preparados y resultados externos; no mover |
| flir-audit-v1-review, FLIR_splits_para_revision | Revisión/histórico potencial; conservar referencias |
| calibration_temporal_v1, calibration_v1, singleton_rank1 | Procedencia no verificada; no mover ni borrar |
| nh-test-tmp-20261015, nh-tmp-final-20261015, nh-tmp-reg-20261015 | GENERATED potencial; no asumir redundancia ni borrar |
| Carpetas con nombres hexadecimales y otras personales | Pertenencia no establecida; fuera del alcance |

Se prepararon directorios vacíos bajo `%USERPROFILE%/Research/FLIR/`:
`code`, `data`, `cache`, `prepared`, `results`, `archive`. No se trasladaron originales,
resultados ni worktrees. La transición debe usar rutas configuradas y, para Git,
`git worktree move` con procesos detenidos y referencias verificadas; no mover
directorios administrados a mano. `.env` permanece privado y sin cambios.

## Validación

Pruebas focalizadas: **99 aprobadas**. Ruff global: PASS. BASE: **1.135 pares**;
TAR: **2.283 miembros comprobados**. Repetir la preparación comprobó los mismos
2.283 archivos sin modificar sus tiempos de escritura. El primer intento de pytest usó un cwd
incorrecto (8 fallos/5 errores por configuración ausente); la repetición desde la
raíz integradora pasó. Frontend compiló; lint sin errores y con tres advertencias
preexistentes. El primer unit test reveló la dependencia privada ya corregida.
La repetición frontend pasó: **94 pruebas**, TypeScript aprobado.
La suite global se registra en el cierre de esta entrega.
No se ejecutaron E2E, Docker, entrenamiento, inferencia, preflight CUDA real ni
evaluación de test. El paso siguiente es revisión de anotaciones y geometría,
no clustering adicional.
