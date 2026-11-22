# Caso FAC: ejecución del pipeline general — 2026-11-06

Publicación local nueva: `operations/jp-general-pipeline-closure-20261106`, bajo
la raíz privada de datos. La entrega vigente es `fac-final/`; los ensayos previos
se conservan con sus snapshots de código. No se sobrescriben las entregas antiguas.

## Población y evidencia observadas

El adaptador consume el inventario existente y el sidecar de localización visual,
valida recibos y conserva **3.377 ocurrencias**: 2.232 primarias, 1.116 referencias
históricas y 29 diagnósticos. Las 1.145 no primarias permanecen explícitamente
excluidas de la propuesta. Las reglas de reserva/QA previas excluyen otras 832
ocurrencias primarias. La población común de ambas políticas es **1.400 ocurrencias**
(700 pares candidatos), con 1.977 excluidas conservadas en el manifiesto.

El catálogo configurable conserva IDs/nombres de origen, incluido `SDZI`, sin
renumerar ni reinterpretar su significado. Las etiquetas se reparsean desde los
TAR originales y conservan bytes, valores y orden. Los TAR y vídeos se rehashean;
no se extraen masivamente ni se modifican.

Se importan **1.116 relaciones candidatas** entre original y variante procesada.
Hay **cero dependencias aceptadas actuales**. Las 12.737 aristas históricas de
recuperación tienen extremos externos a este manifiesto; quedan no evaluables,
con archivo fuente y checksum conservados, sin inventar esos contenidos.
Los grupos históricos se preservan como metadata auxiliar, sin cierre automático.

## Temporalidad

Se reutilizó la evidencia previa del escaneo completo de **289.331 fotogramas**,
sin volver a decodificar los vídeos FAC durante este cierre. Los hashes de sus
fuentes y del sidecar se verificaron nuevamente.

Los 1.116 originales tienen apariciones exactas: **953 únicas y 163 múltiples**,
estas últimas con ocho apariciones cada una. El manifiesto conserva 2.257
posiciones observadas. Las otras 2.261 ocurrencias no reciben PTS confirmado:
incluyen variantes procesadas, referencias y diagnósticos. No se hereda tiempo
desde la imagen original a la variante procesada. El HTML compara el histórico
y ambas políticas en las cinco semillas sobre las posiciones realmente observadas.

## Comparación observada

Ratios explícitos 0,70/0,15/0,15; semillas 0–4; mismo MILP y objetivo en ambas
políticas. Los diez resultados tienen incumbents factibles verificados.
**Las membresías de ambas políticas son idénticas para cada semilla.**
No se fuerza una diferencia ni se afirma beneficio de agrupamiento sin aristas
aceptadas.

| Semilla | Cruces candidatos / pares evaluables | Status solver | Gap |
|---:|---:|---:|---:|
| 0 | 72 / 700 | 0 | 0 |
| 1 | 80 / 700 | 0 | 0 |
| 2 | 81 / 700 | 0 | 0 |
| 3 | 64 / 700 | 4, incumbent factible | 0,0288593 |
| 4 | 70 / 700 | 4, incumbent factible | 0,0117695 |

Los gaps no se ocultan: las semillas 3–4 no se presentan como óptimos probados.
Los cruces corresponden a candidatos estructurales, **no a leakage confirmado**.
Los otros 416 pares no son evaluables en la población asignada. Para seed 0:
980/210/210 ocurrencias en train/val/test.

Los cruces exactos de las propuestas son cero. Los cruces aceptados son cero
con denominador cero: esto no demuestra ausencia de dependencias. La referencia
histórica disponible cubre las 2.232 primarias, sin duplicados exactos en ese
subconjunto. Este universo **no es** el candidato histórico de 1.657 registros;
no contradice ni reemplaza su auditoría de duplicados.

## Auditoría retrospectiva adicional

El [reporte de cierre científico](SCIENTIFIC_CLOSURE.md) revalida por separado el
candidato histórico: 1.657 ocurrencias, 1.459 contenidos, 198 grupos exactos
cross-split (57 train–val, 141 train–test). Documenta ocho grupos con diferencias
de bytes en labels, candidatos, revisiones ambiguas y falta de cobertura.
No aplica las propuestas de la entrega posterior al universo histórico ni
interpreta su igualdad como evidencia de ausencia de leakage. El replay FAC
independiente se repitió durante el congelamiento y aprobó.

## Validación y alcance

Replay CLI independiente: mismas membresías y análisis estable, con relectura de
imágenes/etiquetas y rehash de 13 archivos fuente declarados. La misma población
se reconstruye sin tomar nombres nominales como orden temporal. Se conservan las
exclusiones por conflicto de anotación y reservas; no se modifica BASE, el test
protegido, ni splits previamente publicados.

El replay también pasó desde un entorno nuevo instalado offline desde `uv.lock`.
Se verificó la conservación de SHA256 de 184 archivos de las tres publicaciones
anteriores. La versión HTML final se verificó estáticamente; la revisión en Arc
de la versión anterior motivó la corrección del ancho de tablas y los ticks del
eje, sin cambios en las membresías.

No hubo entrenamiento, features reales nuevas, selección de un test definitivo,
comparación detector ni evaluación causal. Las limitaciones y bloqueos están en
`LIMITATIONS.md`; los comandos reproducibles en `REPRODUCE.md`.
