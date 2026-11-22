# Recuperación de anotaciones NO-HUD

El BASE enviado por el usuario a Hypatia (job 791494) queda intacto. Su envío es
información del operador; no se consultó el estado remoto ni se infieren resultados.

## Resultado local observado (2026-11-04)

Se evaluaron las 950 fuentes: 2.624 pares, incluidos los candidatos existentes de
584 fuentes y alternativas recuperadas para las otras 366. Se reutilizaron 13
registros geométricos. 2.600 pares fallaron el filtro; 24 pares de ocho fuentes
lo superaron. Seleccionar una propuesta por fuente deja 8 imágenes / 12 cajas
(7 train, 1 val). Una fuente queda rechazada por ambigüedad: siete fuentes / diez
cajas permanecen pendientes, no verificadas (6 train, 1 val).

| Clase | Cajas candidatas train tras ambigüedad | Val | Verificadas |
| --- | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 3 | 0 | 0 |
| 2 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 |
| 4 | 6 | 1 | 0 |

Todas las fuentes supervivientes son Adendo. Los ocho destinos seleccionados no
se repiten entre fuentes/splits. Siete pares comparten vecino nativo principal;
uno difiere en un frame. Son resultados de recuperación, no identidad certificada.
No quedaron hipótesis nativas de score >=0,95 sin examinar; esto no prueba búsqueda
exhaustiva bajo otras representaciones o transformaciones.

La galería contiene ocho candidatos geométricos y seis descartes que completan
ejemplos de todas las clases en ambos splits. Inspección visual del asistente:
las ocho escenas candidatas son plausibles, pero los paneles no resuelven identidad
de frames adyacentes ni validan los píxeles sintetizados bajo el HUD. Los seis
descartes muestran escenas/objetos diferentes. No hay adjudicación humana.

**Cero transferencias verificadas, cero etiquetas YOLO admitidas y cero población
emparejada entrenable.** No se generaron paquetes de entrenamiento ni jobs nuevos.
Validación: 31 pruebas offline, Ruff y CLI de ambos scripts aprobados; recibos del
BASE y de los resultados comprobados. No se accedió a payloads del test protegido.

## Procedencia investigada

El usuario identificó el repositorio de su compañera como origen del procesamiento.
Se inspeccionó la revisión
[`dfa52d1`](https://github.com/Laura-Martinez-Galindo/proyecto-FAC/tree/dfa52d1df4f7cca9dbe860fbab689f29ddc90cb3).
Su [configuración](https://github.com/Laura-Martinez-Galindo/proyecto-FAC/blob/dfa52d1df4f7cca9dbe860fbab689f29ddc90cb3/config/videos.json)
declara extracción de 19.800 frames a 30 FPS durante 660 s y procesamiento con
ProPainter de 1920x1080 a 960x540. El
[código de limpieza](https://github.com/Laura-Martinez-Galindo/proyecto-FAC/blob/dfa52d1df4f7cca9dbe860fbab689f29ddc90cb3/scripts/limpiar_hud.py)
mantiene el nombre de entrada al escribir cada resultado de inpainting por bloques.
Esto respalda una variante procesada, no una captura originalmente sin HUD.

No se encontró una vinculación por hashes entre ejecución, máscaras, vídeo y PNG
locales. La revisión actual del código tampoco prueba cuál se ejecutó originalmente.
La condición experimental debe declarar ProPainter + reescalado; no atribuir sus
efectos exclusivamente al HUD. No se ejecutó código externo ni se generó inpainting.

El constructor de datasets publicado compara PSNR y copia etiquetas normalizadas.
Su existencia y el conteo de otro dataset no certifican nuestra población ni la
validez de objetos modificados. No se importan sus splits o ejemplos de test.

## Procedimiento reproducible

`scripts/expand_no_hud_transfer.py` consume el paquete BASE en modo lectura,
publicación de candidatos, revisión previa, índice nativo del vídeo, piloto previo,
ZIPs y un directorio externo nuevo. Comprueba recibos y hashes; reutiliza los
registros previos cuando coincide el par. Evalúa hasta tres candidatos por fuente.
Para fuentes sin candidato invierte una relación de índices ajustada en anclas y
evaluada en anclas reservadas, solamente como hipótesis de recuperación.
No hay offsets, nombres de vídeo ni índices específicos codificados.

Se registran geometría sobre regiones reservadas, cajas transformadas sin clipping,
errores dentro de cada caja sin ocultar regiones por su residual, identidad de
contenido separada por raster y motivos de bloqueo. Las cajas permanecen en JSON.
`scripts/refine_no_hud_evidence.py` busca independientemente vecinos nativos del
destino, comprueba ambigüedad y destinos compartidos entre splits. Ni un vecino
coincidente ni un PSNR alto autorizan entrenamiento.

```powershell
uv run --offline --no-sync python scripts/expand_no_hud_transfer.py --help
uv run --offline --no-sync python scripts/refine_no_hud_evidence.py --help
```

Para admitir pares faltan el mapa fuente FAC -> frame nativo, evidencia de la
ejecución que produjo cada PNG y sus máscaras, y revisión de objetos/cajas en
regiones modificadas. Un segundo entrenamiento solo será admisible con población
verificada, cobertura suficiente y BASE/variante procesada emparejados. No se
prepara un comando sbatch que omita estas condiciones.
