# Línea temporal restringida a vídeos verificados

`/organization/sequences` abre S2-3 y admite únicamente apariciones canónicas
verificadas con los tres vídeos autorizados. Mount activo:
`/runtime/verified-video-timeline-v1`. Las familias de nombres de la proyección
anterior no se dibujan como vídeos. Temporal evidence y linkage se conservan.

## Estado observado

Se preservan las 1.657 ocurrencias S2-3: 1.049 train, 95 validation, 372 test y
141 quarantine. El dataset portátil A conserva train/val sin imágenes ni labels
de test. Es un **candidato de mitigación parcial**, sin restricción por vídeo.
La admisión conserva membership_id y enlaza package_id del dataset entregado.

Las queries localizadas anteriormente no comparten contenido codificado ni RGB
completo con los 1.459 contenidos canónicos. Se admiten **cero** apariciones y
se documentan 1.657 exclusiones. Esto no demuestra ausencia en los vídeos: no
se ha realizado una nueva búsqueda exhaustiva con las queries canónicas.
Los candidatos perceptuales anteriores siguen siendo candidatos.

La entrega B sigue bloqueada por correspondencias y unidades de secuencia
insuficientes. Cruces de secuencia y proximidad en tiempo verificado son `null`,
nunca cero. Los índices nominales son diagnósticos auxiliares, no tiempo real.
PROPOSED es el estado declarado en la publicación fuente, no una nueva decisión.

No hay cluster_id real compatible en los grupos nativos S2-3: se muestra
**Clúster no disponible**. No se reutilizan clusters de otra publicación como
si hubiesen generado esta entrega. group_id no sustituye cluster_id.

## Contrato y admisión

El export anterior `scientific-timeline-v1` permanece intacto como fuente
intermedia. Sus familias inferidas no son apariciones verificadas.
`tools/export_verified_video_timeline.py` consume únicamente esa proyección,
video_correspondence.csv, identidades y hashes de los vídeos, resumen de la
comprobación RGB y receipt/audit de la entrega A. No abre imágenes, vídeos,
labels, pesos ni resultados de detector; no ejecuta ciencia ni genera splits.

Exige identidad frame/content coincidente, método exacto RGB de una búsqueda
completa preexistente, aparición única, SHA-256 de vídeo autorizado y ordinal
decodificado válido. Las asociaciones candidatas/ambiguas/desconocidas quedan
fuera. Expandir basename a nombre con extensión solo cambia presentación tras
verificar frame/content; no establece pertenencia a un vídeo.

Los 15 runs históricos S1/S2/S3 y semillas 0–4 permanecen separados. Cambiar de
run compara membresías existentes y señala que no es el paquete A seleccionado.
No cambia el test ni crea entrega B. Expanded queda fuera de esta proyección.

Se producen index.json, admission.json, coverage.json, excluded_frames.csv y
receipt.json. El CSV offline corresponde a S2-3; el navegador exporta solo el
run seleccionado, separando referencia de correspondencia y de asignación.

Zod valida ambos contratos. El adaptador verifica SHA-256 de index/admission,
procedencia, vídeos, posiciones, contenidos y membresía entregada. Rechaza
índices inferidos y apariciones ambiguas. No hay fallback DEMO. El runtime está
ignorado por Git y excluido de producción con build.copyPublicDir: false.

## Reproducción

Python estándar, sin dependencias nuevas para este adaptador:

```powershell
python tools/export_verified_video_timeline.py PROYECCION_ANTERIOR DIRECTORIO_EVIDENCIA ENTREGA_A DIRECTORIO_NUEVO
```

El destino debe ser nuevo. Montarlo localmente es una operación explícita;
no sobrescribir publicaciones anteriores. La preparación de imágenes del ZIP
se ejecuta fuera de este repositorio, con su operador reproducible independiente.

## Presentación y pruebas

Con posiciones admitidas, frames, intervalos documentados y splits comparten
coordenadas, zoom y desplazamiento. Clusters y secuencias son distintos. Un
cambio de color no crea secuencia; recurrencia no demuestra continuidad. Se
conserva cada frame_id duplicado y hay alternativa de teclado al gráfico.

Con cobertura cero siguen visibles los tres vídeos y motivos de exclusión,
sin inventar marcas, índices o clústeres. Las miniaturas existentes solo se
referencian. Los fixtures positivos son explícitamente sintéticos, nunca
resultados FAC. La comprobación real opt-in valida S2-3 y las exclusiones:

```powershell
$env:FLIR_SCIENTIFIC_TIMELINE_REAL = '1'
npx playwright test e2e/scientific-timeline.spec.ts
```

No se ejecuta entrenamiento, clustering, features ni SLURM. Las comprobaciones
de accesibilidad no certifican WCAG. Los fallos globales preexistentes deben
reportarse separadamente; esta entrega no permite declarar CI verde.
