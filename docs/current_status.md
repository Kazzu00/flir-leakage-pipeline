# Estado actual

## Recuperación de ejecución — 2026-11-22

Corregido el contrato de conteos OE4: se lee el records.parquet enriquecido por
el write_view histórico, sin exigir num_objects al manifiesto fuente. Preparación
CPU con loader Ultralytics real antes de GPU. Replay local real: 1.144 registros,
3.032 objetos, invocación interceptada sin entrenamiento. Ambos brazos denoised
exploratorios preparados sobre 466 train / 47 val tras excluir grupos de riesgo
nominal; 342 correspondencias no resueltas, evaluación común prohibida.
No se ejecutó entrenamiento ni inferencia. [Comandos y límites](runbooks/oe4_recovery_20261122.md).

## Hotfix OE4 — 2026-11-22

OE4 usa `oe4_closeout_v2` y registra la autorización de JP comunicada por el
responsable, sin exigir firmas SSH. Registro explícitamente no criptográfico ni
documentalmente verificado. Freeze y salidas nuevos; integridad, datasets,
checkpoints y requisito de los tres AUG_OFF antes de evaluar conservados.
El usuario reportó preflight remoto previo READY (808314, exit 0); no se ejecutó
entrenamiento ni inferencia local para este hotfix. Véanse los
[comandos de transferencia, preflight y train](runbooks/oe4_hotfix_20261122.md).

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
