# Limitaciones y bloqueos experimentales

El software general puede producir y verificar propuestas sin que FAC tenga
decisiones humanas pendientes resueltas. Eso no convierte la propuesta en un
split definitivo ni en un protocolo detector aprobado.

1. **Aceptación de dependencias FAC:** cero adjudicaciones actuales importadas.
   Las variantes original/procesada siguen candidatas; los grupos históricos
   no se convierten en restricciones nuevas. Las políticas equivalentes no
   demuestran independencia ni beneficio del agrupamiento.
2. **Test protegido:** pendiente de aceptación y congelación legítima común.
   Las reservas existentes se conservan mediante exclusión; no se reasignan.
   La propuesta nueva no autoriza entrenamiento ni modifica BASE.
3. **Correspondencia de variantes:** la igualdad geométrica o los nombres pareados
   no transfieren PTS confirmados a la variante procesada. Se requiere validación
   explícita para tratarla como variante aceptada o ubicarla temporalmente.
4. **Temporalidad:** las apariciones exactas múltiples no resuelven el origen único.
   Sin PTS confirmado no hay eje temporal observado. Ausencia de match exacto
   tampoco demuestra ausencia visual bajo compresión/transformaciones.
5. **Cobertura de evidencia:** las aristas con extremos fuera de la población
   canónica quedan fuera de las métricas del grafo; el adaptador FAC informa su
   número y el checksum del archivo original. No se inventan contenidos externos.
6. **Features:** no hay nuevas extracciones CLIP/DINOv2 en este cierre ni un cálculo
   residual real sobre la nueva población. El consumidor opcional se verifica
   con features sintéticas. No se reutilizan matrices con identidades incompatibles.
7. **Optimización:** un incumbent factible puede tener gap residual. Se preserva
   esa incertidumbre; proporciones/clases son objetivos blandos y no garantizan
   cobertura de clases por split. El replay está condicionado a versiones del
   solver/entorno y bytes de código registrados; no se promete igualdad numérica
   entre versiones arbitrarias.
8. **Validez externa y causal:** tests sintéticos validan contratos del software,
   no exactitud poblacional del alineamiento ni reducción causal del optimismo
   del detector. Cero cruces aceptados solo valida las restricciones declaradas.

Los bloqueos 1–3 pertenecen a la evaluación científica FAC. No bloquean la
ingesta sin labels, el grafo general, las publicaciones sintéticas o el replay.
No se entrenaron modelos, no se modificaron anotaciones y no se ejecutaron
deployment ni monitoring productivos.

La regresión global del repositorio conserva 34 fallos preexistentes del export
organizativo, reproducidos también en el control con solver anterior. El nuevo
componente pasa sus pruebas y replay, pero esto no habilita afirmar que todo el
repositorio tenga CI verde. La corrección de ese contrato local queda separada
de los bloqueos científicos y del alcance de esta generalización.
