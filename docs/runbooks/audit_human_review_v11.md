# FLIR Human Review v1.1

Complementa el [runbook v1](audit_human_review.md). Es infraestructura de auditoría;
no valida una campaña científica ni crea grupos o splits.

## Decisiones y evidencia

Una decisión `reviewed` es conclusiva si continuidad o dependencia no son
`uncertain`, o confianza no es `undetermined`. Requiere al menos una referencia
consultada válida. Se permiten pendientes y revisiones completamente indeterminadas
sin evidencia. El servidor verifica pertenencia al evento, bytes e integridad;
la UI exige marcar explícitamente la evidencia consultada.

Las revisiones v1 se conservan con su política original. Las antiguas conclusiones
sin evidencia aparecen como advertencias y requieren una nueva revisión; no se
reescriben. `verify.valid` verifica integridad estructural; el indicador separado
`latest_reviews_evidence_valid` no certifica calidad científica.

## Asociación visual acotada

1. Seleccionar un candidato y localizar una imagen existente con referencia
   explícita al registro, o realizar correspondencia manual documentada.
2. Comprobar la fuente, el contexto y la identidad del caso. La proximidad temporal
   por sí sola no establece correspondencia.
3. Registrar una imagen PNG/JPEG individual, con localizador verificable,
   responsable y justificación:

```powershell
uv run flir-pipeline sequences audit attach-evidence EVENT_ID evidence.png --db reports/audit-local/audit.sqlite3 --locator "archive/member.png" --basis explicit_record_reference --rationale "Explicit record reference checked" --reviewer "Reviewer"
```

También existe `--basis manual_correspondence`. Se congelan bytes, SHA-256,
asociación al evento y metadata. Límite: 16 MiB y 40 millones de píxeles por imagen.
El identificador permite detectar alteraciones de metadata; no es una firma digital.
La correspondencia declarada exige revisión humana: registrarla no prueba identidad.
La UI separa evidencia registrada de ausencia de evidencia, y ofrece filtros,
orden temporal por fuente y navegación al siguiente pendiente. No hay extracción
masiva ni asociación automática. No versionar imágenes o registros reales.

## Fuente canónica y compañeros

La primera fuente aceptada conserva los IDs canónicos. Si CSV y JSON llegan
juntos, se procesa JSON primero. Si CSV llega antes, su identidad permanece al
incorporar JSON. Se comparan todos los registros antes de aceptar un compañero
del mismo nombre base y directorio original. La fuente adicional queda congelada
con enlaces explícitos; no añade eventos. JSON puede complementar la proyección
temporal, con procedencia explícita, sin cambiar revisiones ni registros originales.

Conflictos de contenido o cambios de bytes de una fuente ya importada se rechazan.
Una revisión distinta requiere otro ledger. Para reconstruir exactamente identidades,
conservar el ledger completo o reproducir el orden original de incorporación.
No se trasladan revisiones automáticamente entre identidades diferentes.

Los duplicados ya creados por v1 se detectan. Su conciliación es explícita:

```powershell
uv run flir-pipeline sequences audit reconcile-companions CANONICAL_ASSET_ID COMPANION_ASSET_ID --db reports/audit-local/audit.sqlite3 --reviewer "Reviewer" --rationale "Equivalent companion records checked"
```

La operación comprueba equivalencia y registra un recibo; conserva físicamente
todos los registros y revisiones. Las historias del compañero se muestran separadas.
No se permiten nuevas revisiones sobre su identidad retirada de la vista activa.
Los adaptadores `legacy_*` reproducen v1 para verificar sus filas congeladas.

## Temporalidad, anomalías y contratos

Se conservan ambos índices de muestra y el tiempo representativo. Si la metadata
declara origen y FPS compatibles con el inicio nominal reportado, los extremos son
`origen + índice / FPS` (índices base cero). Sin metadata suficiente, los extremos
quedan desconocidos; una contradicción explícita con el inicio reportado rechaza
la importación como grilla ambigua. La duración de una muestra no representa precisión
de alineación: `precision_seconds` permanece desconocida.

Un índice compartido entre train/val/test produce una anomalía estructurada con
todas las ocurrencias. No modifica particiones ni confirma leakage semántico.

Los esquemas Draft 2020-12 en `docs/contracts` tienen raíces completas para snapshot,
solicitud de decisión, revisión guardada, exportación e imagen externa. Usan `$defs`
locales, sin acceso a red. Se generan mediante `audit.contracts.write_schemas`.
El identificador sigue siendo `audit-evidence-v1`, con revisión contractual `1.1`.

Consulte [resultados y limitaciones de validación](../analysis/audit_human_review_v11_validation.md).
