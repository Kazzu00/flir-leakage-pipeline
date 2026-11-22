# Estado actual

## Adaptador denoised / OE4 — 2026-11-21

Auditoría real local del TAR: 1.116 pares y filas, imágenes decodificadas,
metadatos originales preservados. Test común local: 372 ocurrencias verificadas,
materializadas sin liberar evaluación. DINOv2/CLIP completos para 1.116 contenidos
y split denoised candidato materializado: 777/172/167, 93 grupos, cero cruces de
1.170 relaciones registradas. MILP factible, óptimo no demostrado; sin freeze
científico ni anclaje verificado al test protegido. Replay de asignación exacto.
Implementados adapters, tests y paquete de código para preflight/AUG_OFF/evaluación
OE4; no GPU ni resultados remotos nuevos. Véanse
[resultados, restricciones y comandos](runbooks/oe4_closeout_20261120.md).

La consolidación de código se describe en [integración final](final_integration.md).
Los registros fechados permanecen en [status](status.md), sin reescribir su
historia. El cumplimiento académico se separa en
[trazabilidad de objetivos](methodology_traceability.md).

El checkout es una candidata local de integración. No implica aceptación de
experimentos, autorización del test, CI verde ni publicación.
