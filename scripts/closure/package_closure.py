"""Seal evidence from completed closure evaluations without changing prior files."""

from __future__ import annotations

import argparse
import ast
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from flir_pipeline.general.publication import read_json, write_json
from flir_pipeline.utils.hashing import sha256_file


def package(repo: Path, data: Path, prior: Path, root: Path):
    comparison = read_json(root / "regression-final/comparison.json")
    verification = read_json(root / "verification/completed.json")
    matrix = read_json(root / "regression-final/failure_matrix.json")
    benchmark = read_json(root / "benchmark/benchmark_summary.json")
    if not (
        comparison["all_34_classified_preexisting"]
        and comparison["same_test_ids"]
        and not comparison["state_differences"]
        and comparison["previous_failed"] == comparison["candidate_failed"] == 34
        and comparison["previous_passed"] == comparison["candidate_passed"] == 55
        and verification["verified"]
        and len(matrix) == 34
        and all(r["replay_verified"] for r in benchmark)
    ):
        raise ValueError("Observed evidence does not support this closure report")
    for receipt, base in (
        (read_json(prior / "previous_publications_sha256.json"), data / "operations"),
        (read_json(prior / "release_receipt.json")["files_sha256"], prior),
    ):
        if any(sha256_file(base / name) != value for name, value in receipt.items()):
            raise ValueError("Previous publication changed")
    output = root / "sealed-final"
    output.mkdir(exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for directory in (
        "src",
        "tests",
        "scripts",
        "configs",
        "docs",
        "exports/frontend/organization/schema",
    ):
        shutil.copytree(
            repo / directory,
            snapshot / directory,
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache"
            ),
        )
    for path in [
        *repo.glob("*.md"),
        repo / "pyproject.toml",
        repo / "uv.lock",
        repo / ".gitignore",
    ]:
        shutil.copy2(path, snapshot / path.name)
    arm_hashes = {}
    for arm in ("previous", "candidate"):
        source = root / "regression-final" / arm
        arm_hashes[arm] = {
            p.relative_to(source).as_posix(): sha256_file(p)
            for p in source.rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".toml", ".lock", ".json")
            and not any(x in p.parts for x in ("__pycache__", ".pytest_cache"))
        }
    differences = sorted(
        name
        for name in arm_hashes["candidate"]
        if arm_hashes["candidate"][name] != arm_hashes["previous"].get(name)
    )
    expected = [
        "src/flir_pipeline/cli.py",
        "src/flir_pipeline/splitting/construction.py",
    ]
    if differences != expected or set(arm_hashes["previous"]) != set(
        arm_hashes["candidate"]
    ):
        raise ValueError("Paired controls differ beyond the declared integration files")
    origin = read_json(root / "regression-final/previous-construction-origin.json")
    committed = subprocess.check_output(
        ["git", "show", f"{origin['commit']}:{origin['path']}"], cwd=repo, text=True
    )
    control = (root / "regression-final/previous" / origin["path"]).read_text(
        encoding="utf-8"
    )

    def milp_ast(source):
        return ast.dump(
            next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "milp_assignment"
            )
        )

    if milp_ast(committed) != milp_ast(control):
        raise ValueError("Previous MILP does not match its declared Git origin")
    write_json(
        output / "comparison_input_hashes.json",
        {
            "arms": arm_hashes,
            "different_files": differences,
            "previous_milp_ast_matches_git": True,
            "previous_origin": origin,
        },
    )
    suite = next(ET.parse(root / "verification/pertinent-tests.xml").iter("testsuite"))
    if (
        suite.attrib["tests"] != "77"
        or suite.attrib["failures"] != "0"
        or suite.attrib["errors"] != "0"
    ):
        raise ValueError("Pertinent regression evidence differs from report")
    git = {}
    for name, argv in {
        "branch": ["branch", "--show-current"],
        "head": ["rev-parse", "HEAD"],
        "status": ["status", "--short"],
        "diff_stat": ["diff", "--stat"],
        "cached_diff": ["diff", "--cached", "--stat"],
    }.items():
        git[name] = subprocess.check_output(["git", *argv], cwd=repo, text=True).strip()
    if git["branch"] != "feat/general-pipeline-closure-20261106":
        raise ValueError("Unexpected branch")
    write_json(output / "git_state.json", git)
    criteria = {
        "frozen_product_sources": "validated_175_unchanged",
        "previous_184_artifacts": "validated_unchanged",
        "prior_release_221_files": "validated_unchanged",
        "organization_regression": "34_preexisting_failures_in_both_controls_no_new_state_differences",
        "global_ci_green": False,
        "technical_general_validation": "77_passed_ruff_cli_and_independent_replays_passed",
        "synthetic_controlled_benchmark": "30_feasible_assignments_atomicity_candidates_unknown_and_reproducibility_checked",
        "historical_fac_exact_overlap": "198_cross_split_content_groups_396_occurrences",
        "fac_nonexact_dependencies": "pending_current_adjudication_and_coverage",
        "annotations": "pending_validation_and_conflict_resolution",
        "detector_effect": "not_evaluated_no_training",
        "protected_test_changed": False,
        "commit_push_merge": False,
        "closure_decision": "bounded_functional_validation_complete_global_stability_and_detector_scientific_conclusion_pending",
    }
    write_json(output / "criteria.json", criteria)
    population_checks = []
    for case in benchmark:
        case_root = root / "benchmark" / case["case"]
        analysis = read_json(case_root / "publication/analysis.json")
        truth = read_json(case_root / "inputs/truth.json")
        populations = [set(run["assignments"]) for run in analysis["runs"]]
        if not all(pop == populations[0] for pop in populations):
            raise ValueError("Synthetic policies or seeds use different populations")
        eligible = populations[0]
        covered = [
            (a, b) for a, b in truth["true_edges"] if a in eligible and b in eligible
        ]
        historical = analysis["historical_assignments"]
        population_checks.append(
            {
                "case": case["case"],
                "same_population_all_policies_and_seeds": True,
                "eligible_occurrences": len(eligible),
                "historical_true_crossings_on_common_eligible_population": sum(
                    historical[a] != historical[b] for a, b in covered
                ),
                "evaluable_true_relations": len(covered),
                "unknown_true_relations": len(truth["true_edges"]) - len(covered),
            }
        )
    write_json(output / "benchmark_population_control.json", population_checks)
    report = (repo / "SCIENTIFIC_CLOSURE.md").read_text(encoding="utf-8")
    report += "\n## Matriz individual de regresión\n\n"
    report += "Todas las filas remiten a los testcase de previous.xml y candidate.xml. "
    report += "La matriz CSV/JSON conserva causa, ubicación, afectación, acción y aserciones bloqueadas.\n\n"
    report += "| Prueba | Anterior | Candidato | Clasificación |\n|---|---|---|---|\n"
    for row in matrix:
        report += f"| `{row['test_id']}` | {row['previous']} | {row['candidate']} | {row['classification']} |\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    (root / "README.md").write_text(
        "# Paquete de congelamiento científico\n\n"
        "Reporte: sealed-final/report.md. Cumplimiento: sealed-final/criteria.json. "
        "Reproducción: sealed-final/source_snapshot/REPRODUCE.md.\n\n"
        "- regression-final/: comparación válida, matriz, JUnit, entorno y fuentes de ambos brazos.\n"
        "- regression-validated/: control CLI anterior insuficiente para aislar el solver.\n"
        "- sealed/: intento incompleto de sellado; detectó la limitación del control anterior.\n"
        "- regression/: intento inicial inválido de setup, si existe; no respalda conclusiones.\n"
        "- benchmark/: protocolo, verdad sintética, inputs, publicaciones y controles.\n"
        "- fac-audit/: retrospectiva separada por población y recibos de entradas.\n"
        "- verification/: 77 tests, Ruff, CLI, integridad y replays independientes.\n"
        "- sealed/: snapshot reproducible, evidencia Git y reporte final.\n\n"
        "CI global no verde. Sin entrenamiento, cambios de test protegido, commit, push o merge. "
        "Contiene evidencia privada; no versionar este directorio.\n",
        encoding="utf-8",
    )
    files = {
        p.relative_to(root).as_posix(): sha256_file(p)
        for p in root.rglob("*")
        if p.is_file()
        and not any(
            x in p.parts for x in ("__pycache__", ".pytest_cache", ".ruff_cache")
        )
    }
    write_json(
        output / "receipt.json",
        {
            "files_sha256": files,
            "excluded": [
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "sealed-final/receipt.json",
            ],
            "criteria": criteria,
            "snapshot_is_dirty_worktree": True,
        },
    )
    if any(sha256_file(root / name) != value for name, value in files.items()):
        raise ValueError("Package changed during sealing")
    print(f"Sealed and reverified {len(files)} files; global CI remains not green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "data", "prior", "package"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    package(*(p.resolve() for p in (args.repo, args.data, args.prior, args.package)))
