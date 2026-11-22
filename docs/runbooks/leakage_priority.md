# Prioridad: experimento por secuencias, NO-HUD cerrado por ahora

## Evidencia recuperada: 2026-11-05

Se verificaron los cuatro checksums de salida del linkage y el hash de las
asignaciones que registra su metadata. Las 9.648 ocurrencias cubren 16 secuencias
y 11 grupos visuales, sin romper dependencias exactas en esa tabla. Seis secuencias
forman un grupo históricamente confirmado; las otras diez son grupos unitarios
sin enlace confirmado, lo que no demuestra independencia. Falta la metadata del
productor VDG y las publicaciones/eventos vinculados de revisión para validar su
procedencia completa. No se reconstruyeron agrupamientos.

De BASE, 902 fuentes tienen candidatos históricos y 48 carecen de linkage; 787
tienen candidatos de grupo compartidos por ambos encoders. Las 1.568 asociaciones
del resumen global se reprodujeron desde las tablas originales. Las decisiones
manuales cubren 12 fuentes BASE: seis supported, cinco ambiguous y una unsupported.
Las 938 restantes no están revisadas, incluidas las 67 de val.

Los seis supported de train pertenecen a dos grupos que también tienen cinco
contenidos supported del test fijo. Esto requiere adjudicación antes de convertir
el soporte histórico en restricciones de partición; no confirma leakage por sí
solo. El enlace histórico video_11min requiere además reconciliar su procedencia.
Las revisiones históricas no sustituyen las decisiones de octubre. Cero nuevas
asignaciones autorizadas, ningún split ni entrenamiento adicional generado.

Reproducción (rutas externas suministradas por el operador; salida nueva):

```powershell
uv run --offline --no-sync python scripts/assess_sequence_evidence.py --evidence $evidence --base $base --fixed-publication $fixedPublication --output $newOutput
```

Siguiente paso metodológico: recuperar metadata del VDG y publicaciones de revisión
vinculadas, reconciliar decisiones de octubre y resolver enlaces pendientes y
conflictos con test. Si el cierre obliga a excluir contenidos, preparar dos brazos
con población común sin modificar el BASE ya enviado. La evidencia local queda
fuera de Git, en summary.json, base_link_assessment.csv e input_checksums.json.

## Registro previo (2026-11-04)

Comprobación acotada del 2026-11-04 sobre las 950 imágenes BASE; sin búsquedas
globales nuevas, nuevo split, entrenamiento ni cambios al job 791494.

| Familia | Train | Val | Procedencia: alcance de la evidencia |
| --- | ---: | ---: | --- |
| video_11min | 415 | 27 | Upstream distingue otra grabación; no asignar al clip NO-HUD por el alias video1 |
| video_13min | 429 | 31 | Hipótesis Codefest; no adoptar sus offsets como verdad |
| scene, 720p | 32 | 8 | Hipótesis de escenas Codefest; nombre no acredita secuencia revisada |
| scene, 1080p | 7 | 1 | Ocho candidatos plausibles al clip de 19.800 PNG |

Revisión upstream fijada en
[`2b35e89`](https://github.com/Laura-Martinez-Galindo/proyecto-FAC/tree/2b35e89cf5cb51841e6319cbc27b8c7c6973d4c8):
`config/videos.json` identifica video1 como el clip de mayo; el exportador asigna
también video_11min a ese alias pese a advertir que es otra grabación. Su parser
numérico toma el prefijo v1 como índice en nombres de escenas. El parser local
nuevo extrae explícitamente frame/scene, sin asignar vídeo, tiempo o secuencia.
También se distingue índice nativo cero-based de nombre PNG uno-based; no se
copian las fórmulas/offsets específicos del exportador al pipeline.

Los siete casos pendientes se contrastaron únicamente con tres frames nativos
alrededor de su segundo nominal, sobre el vídeo ligado por SHA256 al índice previo.
Los siete mínimos de error RGB coinciden con el segundo nominal (MAE 0,86–1,03);
dos difieren en un frame del vecino por descriptor. Ninguno es igualdad raster
exacta. Geometría plausible y paneles previos no prueban extracción ni integridad
de objetos bajo inpainting. **Cero transferencias verificadas**; no producir otro
dataset parcial. Clasificación por imagen, split y clase conservada externamente.

## Comprobaciones del experimento principal

BASE ya conserva S3: 928 grupos, de los cuales 923 son singletons y cinco contienen
27 imágenes. Cero grupos S3 cruzan train/val; cero duplicados exactos train/val y
cero intersección con los 373 contenidos protegidos. Los checksums de los metadatos
de protección/test coinciden con la publicación fija. No se abrió payload de test.
Las cinco clases permanecen en ambos splits; distribución detallada externa.

S3 representa cierre de relaciones candidatas, no cobertura completa de secuencias
extraídas. Repetir S3 no sería un nuevo contraste de leakage. El set de secuencias,
VDGs y linkage remoto tiene rutas históricas aportadas por el usuario; existencia,
integridad y cobertura sobre estas 950 imágenes pendientes de inspección local.
Las muestras de revisión no equivalen por su nombre a decisiones confirmadas.

El siguiente paso es recuperar solo metadata de esos artefactos, conservando las
fuentes remotas y el job BASE. Después: verificar joins content -> sequence -> VDG,
cobertura y atomicidad, dependencias con la reserva fija y soporte de clases.
No agrupar por vídeo completo ni rellenar enlaces desconocidos con singletons.
Si la protección de secuencias exige excluir contenidos, ambos brazos comparables
necesitarán la misma nueva población; no modificar retroactivamente BASE.

Mantener el protocolo congelado (YOLO11m, seed 42, 50 epochs, imgsz 640, batch 1,
nbs 64 y demás parámetros). Cambiar val impide comparar directamente sus métricas;
se requiere una evaluación común fija, sin usar test para elegir hiperparámetros.
No hay nuevo entrenamiento admisible hasta resolver estas condiciones.
