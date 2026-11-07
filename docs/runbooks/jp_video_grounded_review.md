# JP: revisión sustentada en el vídeo original

La corrección metodológica posterior a la entrega del 5 de noviembre invalida
el uso de nombres, índices nominales y tiempos suministrados como localización.
Los manifiestos publicados se preservan como evidencia histórica provisional.
Sus ejes temporales nominales no deben utilizarse para aceptar secuencias,
dependencias ni un split definitivo.

La revisión externa añade un paquete separado: cinco grupos candidatos, nombres
e identidades completos, aristas históricas recuperadas, decisiones vacías
`supported / ambiguous / rejected`, exclusiones de conflictos y validación
estática de HTML/CSV. Esta última no sustituye la inspección manual en navegador.
No se ejecuta automatización del navegador.

## Localización independiente

`alignment/visual_localization.py` contiene primitivas que no reciben índices,
nombres ni tiempos nominales. El ejecutor externo abre los tres vídeos explícitos
en lectura, calcula SHA-256 y decodifica secuencialmente todo su contenido.
Conserva ordinal decodificado, PTS y time base. Un digest de píxeles muestreados
recupera candidatos; únicamente el digest de **todo** el raster RGB, incluyendo
dimensiones, verifica igualdad exacta. Se conservan todas las apariciones.

Una coincidencia exacta única tras completar el recorrido sustenta la verificación
**del original**, con el alcance de esos vídeos. Se contrastan además alternativas
visuales globales para detectar recurrencias transformadas; una alternativa
competitiva mantiene el caso como ambiguo para revisión. Sin alternativas
competitivas detectadas se registra `VIDEO_VERIFIED`. Varias
apariciones exactas son `VIDEO_AMBIGUOUS`. Ausencia de igualdad exacta sigue
siendo `NOT_VERIFIED`: compresión, resolución o transformación pueden impedirla.
`VIDEO_UNMATCHED` exige una búsqueda visual completada y documentada; no equivale
a no encontrar igualdad de píxeles. Si falta PTS, el ordinal real se conserva,
pero no se inventa un tiempo usando FPS.

Un índice de luminancia interior normalizada, muestreado desde PTS reales,
recupera alternativas aproximadas en todos los vídeos. Sus scores son criterios
de recuperación, nunca probabilidades ni aceptación científica. Geometría,
contexto y continuidad se revisan por separado. Se deben conservar las
alternativas y declarar los límites de muestreo y de recuperación.

NO-HUD se vincula primero con su original. Igualdad de nombre/label y pertenencia
a una pareja estructural no certifican conservación visual de objetos. Hasta
validar ese vínculo no se propaga una posición verificada del original a la
pareja completa. Tampoco se deriva continuidad de una localización puntual.

## Reproducción de la separación publicada

El paquete externo conserva código base archivado, overlay de implementación,
lockfile, configuración y un replay que compara en memoria las 2.232 filas con
el CSV publicado byte por byte. Su comando `reproduce.ps1` exige raíz de evidencia
y directorio de trabajo nuevos; no necesita `.git`, `.env` ni productor opcional.
La primera instalación necesita registro de paquetes o caché completa de uv.
Los hashes deterministas de membresía/CSV no incluyen duración, ejecutable ni
metadata de ejecución. No se escriben nuevos manifiestos de split.

No confundir reproducir el algoritmo desde los TAR e inventarios declarados con
recrear el procesamiento que produjo los TAR. El productor no es necesario para
la primera tarea. Sin código/configuración de ese productor no puede certificarse
la regeneración exacta de sus imágenes; el algoritmo consume esas imágenes como
entradas explícitas y verificadas por hash. Su procedencia visual se puede y debe
auditar directamente sin hacer depender el algoritmo de ese productor.

La reproducción limpia valida software y asignaciones provisionales. Para un
split definitivo faltan correspondencias aceptadas, tratamiento explícito de
ambigüedad, continuidad y dependencias, conflictos de anotación resueltos o
excluidos, y decisión sobre la reserva protegida. Una comparación posterior de
modelos requiere además poblaciones/protocolos congelados y aprobación del test.
