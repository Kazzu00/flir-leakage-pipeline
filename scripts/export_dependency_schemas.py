"""Rebuild public pair schemas from executable models; contains no real data."""

import json
from pathlib import Path

from flir_pipeline.alignment.dependency_contract import Bundle
from flir_pipeline.alignment.dependency_store import PairDecision, ReviewExport


def main():
    root = Path(__file__).resolve().parents[1] / "docs/contracts"
    for name, model in (
        ("dependency-pilot-v1", Bundle),
        ("dependency-decision-input-v1", PairDecision),
        ("dependency-review-export-v1", ReviewExport),
    ):
        schema = model.model_json_schema()
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["$id"] = f"https://flir-pipeline.local/contracts/{name}.schema.json"
        if model is PairDecision:
            schema["allOf"] = [
                {
                    "if": {
                        "properties": {"status": {"const": "reviewed"}},
                        "required": ["status"],
                    },
                    "then": {
                        "required": ["reviewer", "justification", "evidence_consulted"],
                        "properties": {
                            "reviewer": {"pattern": "\\S"},
                            "justification": {"pattern": "\\S"},
                            "evidence_consulted": {"minItems": 1, "uniqueItems": True},
                        },
                    },
                }
            ]
        (root / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
