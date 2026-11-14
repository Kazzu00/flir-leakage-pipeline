"""Regenerate ONLY the tiny synthetic exporter/TypeScript contract test fixture."""

import json
from pathlib import Path

from test_exporter import ExportTests

case = ExportTests()
case.setUp()
try:
    case.reduction()
    case.reduction("clip")
    target = Path(__file__).parents[2] / "src/test/fixtures/leakage-synthetic.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(case.exporter.snapshot(), indent=2) + "\n", encoding="utf-8")
finally:
    case.tearDown()
