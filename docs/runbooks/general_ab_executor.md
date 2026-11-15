# Explicit general A/B handoff (integration only)

The scientific pipeline remains frozen at `df3452f`. This integration starts from
the existing common-executor implementation at `6f349c5`, in a separate worktree
and branch. No grouping, solver, MILP objective, detector model, trainer or
scientific publication is changed. The historical fixed-test route remains the
default; the adapter must be requested explicitly.

## Input contract

`--input-format general_ab_assignments_v1` accepts a directory with `bundle.json`:

```json
{
  "schema_version": "general_ab_assignments_v1",
  "dataset_id": "<original SHA256 dataset identity>",
  "population_id": "<original population identity>",
  "test_status": "PROPOSED",
  "classes": {"0":"vehicle","1":"building","2":"road","3":"river","4":"SDZI"},
  "sources": {"images":{"path":"images.zip","sha256":"<sha256>"},
              "labels":{"path":"labels.zip","sha256":"<sha256>"}},
  "records": {"path":"records.csv","sha256":"<sha256>"},
  "test_ids": {"path":"test.csv","sha256":"<sha256>"},
  "protected_contents": {"path":"protected.csv","sha256":"<sha256>"},
  "accepted_groups": {"path":"groups.csv","sha256":"<sha256>"},
  "arms": ["<ten arm objects described below>"]
}
```

Each arm contains `arm_id`, `policy`, integer `seed`, and an `assignments` binding
with relative `path` and `sha256`. Policies are `exact_content_balanced` and
`dependency_group_balanced`, each with split seeds 0–4. IDs are preserved, not
translated into S1/S2/S3 or an expanded population.

CSV contracts:

- `records`: unique `frame_id`, `content_id`, `image_sha256`, `label_sha256`,
  `image_source`, `image_member`, `label_source`, `label_member`. Additional
  occurrence metadata is retained. Exact content identity equals image SHA256.
- `test_ids`: unique `frame_id`; common test membership.
- `protected_contents`: unique `content_id`, including the full accepted-group
  closure of test contents. Other reserved contents may also be protected.
- `accepted_groups`: `content_id,group_id`, exactly once for every content. This
  is an explicit upstream declaration, **not scientific acceptance by the adapter**.
- Each assignment: **only** `frame_id,new_split`, all and only the declared
  occurrences; values `train`, `val`, `test`, `quarantine`. If upstream files
  contain other identity columns, first compare them row-wise with `records`
  and bind that projection to the source file hash. Do not drop them unchecked.

All relative locators are confined to their declared roots. Source ZIP and member
hashes, class IDs, label syntax and image decoding are checked. Labels are copied
byte for byte; the adapter cannot resolve conflicting annotations or class meaning.
The class-4 SDZI/HeavyMachinery interpretation still requires human adjudication.

## Non-training operations

Use the existing Python 3.11 environment (`uv` remains the environment manager).
Choose fresh external outputs, disjoint from source and publication directories:

```powershell
uv run flir-pipeline detection common-executor dry-run $Bundle $Plan --input-format general_ab_assignments_v1 --config configs/detection/common_test_yolo11m_v2.yaml
uv run flir-pipeline detection common-executor prepare $Bundle $SourceRoot $Prepared --input-format general_ab_assignments_v1
uv run flir-pipeline detection common-executor verify $Prepared --kind preparation
```

The preparation copies only train/validation payloads. Test and quarantine remain
metadata; neither appears in the training YAML. Verification checks immutable
memberships, same test, shared excluded population, protected exclusion, exact
indivisibility and policy B's declared accepted-group indivisibility. Policy A
may split accepted nonexact development groups: that is the intended baseline.

Plans deduplicate identical memberships and detector seeds, retaining every policy
alias. The detector configuration is shared; official checkpoint hash and origin
must be frozen later. Source bundle hash participates in plan identity. A proposed
bundle or successful materialization never authorizes training.

## Admission and eventual test release

The existing attributed OpenSSH signatures, trusted signer allowlist, clean
checkout/environment, approved test, reviewed decisions, verified weights,
positive common batch and OS isolation gates remain mandatory. For this format,
the signed admission must explicitly contain `input_format`, original `population`
and `dataset_id`, both long policy names, split seeds 0–4, detector seeds 42–44,
and `max_runs` equal to the number of unique admitted detector identities.
Plan, preparation and test record identity must agree with the signed bindings.

All 30 logical aliases must be represented once. If A and B have identical
memberships for every split seed, admission rejects redundant A/B training.
Do not invent dependency constraints just to obtain a contrast. A genuine
partial contrast retains aliases for the shared identities; those shared runs
are not independent experimental replicates.

Only a separately signed evaluation release, after the whole admitted matrix has
completed, can call the existing custodian test preparer. For A/B its
`test_sources` contains `general_ab_bundle: {path,sha256}` pointing to the same
`bundle.json`, and `source_root`. It rebinds the test to the signed identities,
copies original bytes and uses the existing test-view verifier/evaluator. Test
sources must remain inaccessible to training workers, including review galleries
and archival copies. Local development materialization does not prove OS isolation.

## Validation and limits

Synthetic fixtures exercise both distinct policies, checksums/identities, common
test, quarantine and protected closure, class IDs/labels, deterministic repeated
materialization, detector identity, proposed-test rejection, alias-preserving
signed-context orchestration and released-test verification. Existing common and
scientific executor tests are included. No detector is trained by these tests;
mocked runtime checks are not empirical model results.

Local focused run: 68 tests passed. Global Ruff passed. The runtime preflight is
blocked locally by absent Torch/Torchvision/Ultralytics; interactive SSH is needed
to inspect Hypatia. No GPU, viable batch, real signed worker execution or detector
comparison is validated by this adapter. This is not a claim of green global CI.
