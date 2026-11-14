"""Self-contained document/message schemas; generation uses no remote resolver."""

import json
from pathlib import Path

from .models import (
    DecisionExport,
    DecisionInput,
    ExternalEvidence,
    ReviewDecision,
    Snapshot,
)

DOCUMENTS = {
    "audit-evidence-v1": Snapshot,
    "audit-decision-input-v1": DecisionInput,
    "audit-decision-export-v1": DecisionExport,
    "audit-review-decision-v1": ReviewDecision,
    "audit-external-evidence-v1": ExternalEvidence,
}


def schema_for(name):
    schema = DOCUMENTS[name].model_json_schema(mode="validation")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"urn:flir:{name}:1.1",
        **schema,
    }


def write_schemas(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for name in DOCUMENTS:
        (directory / f"{name}.schema.json").write_text(
            json.dumps(schema_for(name), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
