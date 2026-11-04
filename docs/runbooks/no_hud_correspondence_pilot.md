# Piloto de recuperación de correspondencias HUD/no-HUD

Este piloto investiga por qué la recuperación anterior no produjo identidades
verificadas. No modifica `no_hud_label_transfer_v1`, no copia anotaciones y no
adjudica revisión humana. Los productos son evidencia diagnóstica externa.

## Ejecución reproducible

El decoder se utiliza como dependencia aislada, sin incorporarlo a core/CI:

```powershell
uv run --with av==18.1.0 python scripts/build_no_hud_alignment_pilot.py index-video `
  --video <video-original.mp4> --output <indice-externo-nuevo>

uv run --with av==18.1.0 python scripts/build_no_hud_alignment_pilot.py pilot `
  --publication <publicacion-no-hud-v1> --video-index <indice-externo> `
  --output <piloto-externo-nuevo>

uv run pytest tests/test_no_hud_alignment.py tests/test_no_hud.py `
  tests/test_video_frames.py tests/test_video_manifest.py -q
uv run ruff check .
```

El índice conserva SHA-256 del video, decoder/bibliotecas, metadatos declarados,
índice en orden de presentación, PTS y time base de cada cuadro decodificado.
El descriptor usa RGB 160×90 de PyAV y la misma receta central 32×24 del buscador
anterior. No se presupone tasa de cuadros, offset, origen del contador ni stride.
La igualdad entre conteos de PNG y cuadros no verifica una relación.

## Selección y revisión

Se vuelven a medir las 60 parejas de la galería previa. El piloto selecciona hasta
20 parejas: una por intervalo igual del índice nominal PNG, priorizando el mayor
score entre los tres vecinos previos, con contenido fuente distinto y desviación
de gris diagnóstica >=5. La restricción de contenido distinto impide repetir la
misma fuente. La desviación global **no detectó** una fuente casi uniforme con
bordes contrastantes: sus 18 parejas anteriores pasaron el filtro geométrico.
Un diagnóstico posterior encontró mediana de desviación por tile igual a cero;
no se recalibraron umbrales ni se cambiaron retrospectivamente los resultados.
El filtro no certifica calidad de imagen ni identidad y la selección no es
una muestra representativa de toda la población. No se utilizan clases o splits
para ordenar candidatos; la protección se aplica como metadata posterior.

## Registro y métricas

Se compara resize de cuadro completo, traslación y similitud (traslación, escala
uniforme y giro). Las coordenadas de ajuste son 320×180; los límites son ±12
píxeles, escala .94–1.06 y giro ±3 grados. La parametrización es de muestreo inverso,
no una transformación de cajas YOLO.

El soporte diagnóstico excluye bordes, retícula central y verde dominante dilatado.
**No es una máscara HUD validada**: puede excluir vegetación y dejar modificaciones
desconocidas. El ajuste robusto usa una mitad de los tiles de 16 píxeles; la otra
mitad mide NCC, MAE y RMSE. Todos los modelos se comparan sobre la intersección
común de soporte después de transformar la máscara fuente. Los tiles de evaluación
se usan para comparar modelos, no son un test independiente de generalización.

El filtro provisional exige textura >=5, NCC >=.95 y MAE <=12 niveles de gris,
además de convergencia sin alcanzar límites. Estos valores se fijan para el piloto,
no son umbrales calibrados de identidad. Un candidato que falla queda rechazado
**bajo este protocolo geométrico**, sin afirmar que no exista otro match o que una
transformación fuera de los límites no pueda explicarlo. El registro central no
demuestra ausencia de recorte o distorsión fuera del soporte.

Cada pareja incluye un control con PNG desplazado media colección. Es un contraste
diagnóstico, no un negativo temporal certificado: puede contener escenas repetidas.

## Evidencia temporal

La imagen fuente y el PNG se comparan separadamente con los descriptores de todos
los cuadros nativos. Se conservan los cinco vecinos, márgenes y discrepancias de
índice. Se decodifican ventanas de ±2 alrededor de cada primer vecino (como máximo
200 cuadros seleccionados para 20 parejas), fuera del dataset original. Las
imágenes nativas de revisión son resize 960×540; el hash RGB nativo y PTS se conservan
separadamente. No se extrae el video completo ni se altera el MP4.

Se repite registro y evaluación en cada ventana. Se informa qué alternativas
quedan a <=0.5 niveles de gris del menor MAE: es sensibilidad diagnóstica local,
no una prueba de unicidad global. El vecino de descriptor y el mejor registro
pueden discrepar. Ninguno certifica procedencia de una imagen JPEG con pipeline
desconocido o de un PNG modificado por un proceso no documentado.

Los modelos índice PNG→índice nativo y PNG→PTS se estiman mediante mediana de
pendientes y offset, con anclas alternas para ajuste/evaluación. Las familias de
nombres fuente se analizan separadamente, si tienen suficientes anclas. No se
fuerza pendiente 30 ni offset −1. Se conservan todos los residuos; tampoco se
extrapola un modelo exitoso del piloto a los 19.800 PNG.
En `temporal_fit`, los residuos siguen el orden creciente de la variable de
entrada; `fit_indices` y `held_out_indices` remiten a las posiciones originales
de las anclas entregadas al ajuste. Esto importa para familias no ordenadas por
su índice nominal en la selección del piloto.

## Estados y bloqueo

Cada pareja distingue `visual_candidate`, `registration.estimated`,
`geometry_screen_pass`, `frame_identity_verified` y `human_validated`.
El último permanece falso y no se inventan nombres de revisores humanos.
En este protocolo, compatibilidad geométrica y ajuste temporal son evidencia
candidata. Sin un vínculo nativo verificable y una receta de procesamiento
reproducible, la identidad se mantiene pendiente aunque ambas métricas sean buenas.
No se emiten relaciones `same_native_frame`, anotaciones o asignaciones de split.
Todos los registros mantienen `development_blocked=true`, `ready_for_training=false`.

Para resolver ambigüedades se requiere del productor: hash y stream del video,
decoder/versión, comando y filtros exactos, PTS o índice nativo por archivo,
convención inicial 0/1, regla de muestreo y manejo de cuadros repetidos/descartados,
crop/resize/interpolación/orientación, parámetros y máscara del proceso HUD,
y configuración de color/compresión. También se necesita la procedencia nativa
de las imágenes etiquetadas. Los registros deben ligarse a los SHA-256 de los
archivos actuales; una confirmación general de que proceden del mismo video no
resuelve diferencias entre cuadros consecutivos.
