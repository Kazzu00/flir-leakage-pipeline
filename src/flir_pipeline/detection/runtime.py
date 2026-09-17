"""Optional Ultralytics execution with GPU gate, isolated checkpoints and resume.

Heavy imports are local so core/CI never needs torch, model weights or a GPU.
"""

from __future__ import annotations

import copy
import importlib.metadata
import os
import platform
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flir_pipeline.data.classes import class_name
from flir_pipeline.detection.materialization import verify_view
from flir_pipeline.detection.metrics import (
    ImageStats,
    bootstrap,
    evaluate,
    save_image_stats,
)
from flir_pipeline.detection.protocol import (
    detector_run_id,
    experiment_matrix,
    verify_plan,
)
from flir_pipeline.similarity.storage import (
    execution_provenance,
    file_sha256,
    read_json,
    stable_id,
    write_json,
)


def environment() -> dict:
    import psutil
    import torch
    import ultralytics

    packages = {p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python", "pillow")}
    ram = psutil.virtual_memory()
    return {"packages": packages, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "cpu": platform.processor(), "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
            "ram_total_bytes": ram.total, "ram_available_bytes": ram.available,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
            "gpu_free_bytes": torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None,
            "ultralytics_defaults_sha256": file_sha256(Path(ultralytics.__file__).parent/"cfg/default.yaml")}


def environment_signature(env: dict) -> dict:
    return {k: v for k, v in env.items() if k not in ("ram_available_bytes", "gpu_free_bytes")}


def provenance() -> dict:
    result = execution_provenance()
    result["detection_source_sha256"] = {p.name: file_sha256(p) for p in Path(__file__).parent.glob("*.py")}
    return result


def prepare_weights(output: Path) -> Path:
    path = output/"pretrained"/"yolo11n.pt"
    receipt = path.with_suffix(".json")
    if path.exists():
        if not receipt.exists() or read_json(receipt)["sha256"] != file_sha256(path):
            raise ValueError("Unverified existing pretrained checkpoint; preserved for inspection")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    partial = path.with_suffix(".download.partial")
    if partial.exists():
        raise ValueError("A partial weights download exists; inspect it before retrying")
    with urllib.request.urlopen(url, timeout=60) as source, partial.open("xb") as target:
        while block := source.read(1024*1024):
            target.write(block)
    partial.replace(path)
    write_json(receipt, {"url": url, "model": "yolo11n.pt", "sha256": file_sha256(path)})
    return path


def configure_runtime() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    config_root = Path(".cache/ultralytics").resolve()
    (config_root/"Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_root)
    import torch
    from ultralytics import settings

    # Prevent optional installed integrations from uploading this private dataset.
    settings.update({k: False for k in ("sync", "wandb", "mlflow", "comet", "clearml", "neptune", "dvc", "hub", "tensorboard") if k in settings})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def capture_validator(fixed_confidence: float):
    from ultralytics.models.yolo.detect import DetectionValidator

    class CapturingValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_statistics = []

        def update_metrics(self, preds, batch):
            for i, prediction in enumerate(preds):
                truth = self._prepare_batch(i, batch)
                pred = self._prepare_pred(prediction)
                conf = pred["conf"].cpu().numpy()
                pc = pred["cls"].cpu().numpy().astype(int)
                gt = truth["cls"].cpu().numpy().astype(int)
                matched = self._process_batch(pred, truth)["tp"]
                mask = pred["conf"] >= fixed_confidence
                filtered = {k: v[mask] for k, v in pred.items()}
                fixed_pc = filtered["cls"].cpu().numpy().astype(int)
                fixed_tp = self._process_batch(filtered, truth)["tp"][:, 0]
                self.image_statistics.append(ImageStats(Path(batch["im_file"][i]).stem, conf, pc, matched, gt,
                                                        np.bincount(fixed_pc, minlength=5), np.bincount(fixed_pc[fixed_tp], minlength=5)))
            super().update_metrics(preds, batch)

    return CapturingValidator


def evaluate_checkpoint(checkpoint: Path, dataset: Path, config: dict, output: Path, device: str = "0") -> dict:
    evaluation = config["evaluation"]
    klass = capture_validator(evaluation["precision_recall_confidence"])
    validator = klass(args={"model": str(checkpoint.resolve()), "data": str(dataset.resolve()), "split": "test", "mode": "val", "task": "detect",
                            "imgsz": config["train"]["imgsz"], "batch": config["train"]["batch"], "workers": 0, "device": device,
                            "conf": evaluation["confidence_floor"], "iou": evaluation["nms_iou"], "max_det": evaluation["max_det"],
                            "half": False, "rect": False, "augment": False, "plots": False, "save_json": False, "verbose": False}, save_dir=output/"test_validation")
    validator()
    if validator.names != {i: class_name(i) for i in range(5)}:
        raise ValueError("Checkpoint class mapping differs from canonical vocabulary")
    images = sorted(validator.image_statistics, key=lambda s: s.frame_id)
    save_image_stats(images, output/"test_image_stats.npz")
    overall, classes = evaluate(images)
    # AP parity catches a changed upstream matching/statistics API. Stable confidence
    # ties can differ slightly from upstream quicksort, hence explicit tolerance.
    upstream = float(validator.metrics.box.map)
    if overall["map50_95"] is not None and abs(overall["map50_95"]-upstream) > 1e-3:
        raise ValueError("Captured AP does not agree with pinned Ultralytics within 0.001")
    ci, samples = bootstrap(images, evaluation["bootstrap_resamples"], evaluation["bootstrap_seed"], evaluation["confidence_level"])
    classes.to_parquet(output/"metrics_per_class.parquet", index=False)
    samples.to_parquet(output/"bootstrap_samples.parquet", index=False)
    write_json(output/"bootstrap.json", ci)
    result = {"overall": overall, "definition": evaluation, "upstream_map50_95": upstream,
              "test_threshold_optimized": False, "checkpoint_selection": "validation_only"}
    write_json(output/"metrics.json", result)
    return result


def check_loader_annotations(trainer, expected_counts: dict[str, int]) -> None:
    """Fail if upstream cache/parser silently discards a canonical annotation."""
    for loader in (trainer.train_loader, trainer.test_loader):
        for label in loader.dataset.labels:
            frame = Path(label["im_file"]).stem
            if frame not in expected_counts or len(label["cls"]) != expected_counts[frame]:
                raise ValueError("Detector loader changed canonical annotation support")


def synthetic_smoke(plan_directory: Path, output: Path, gpu: bool) -> dict:
    """Explicit runtime exercise, not a pilot or scientific observation."""
    from PIL import Image, ImageDraw

    configure_runtime()
    env = environment()
    plan = verify_plan(plan_directory)
    config = plan["identity"]["config"]
    output.mkdir(parents=True, exist_ok=True)
    if gpu and not env["cuda_available"]:
        receipt = {"state": "BLOCKED_NO_CUDA", "environment": env, "scientific_training_executed": False}
        write_json(output/"probe.json", receipt)
        return receipt
    import torch
    from ultralytics import YOLO

    weights = prepare_weights(output.parent)
    if (output/"train").exists():
        raise ValueError("Smoke destination already exists; preserve it and use a new destination")
    for split in ("train", "val", "test"):
        for i in range(8):
            image = Image.new("RGB", (96, 96), (20+i*2, 30, 40))
            draw = ImageDraw.Draw(image)
            draw.rectangle((24, 24, 72, 72), fill=(80+i*12, 160, 200))
            ip = output/"synthetic"/"images"/split/f"{split}_{i}.png"
            lp = output/"synthetic"/"labels"/split/f"{split}_{i}.txt"
            ip.parent.mkdir(parents=True, exist_ok=True)
            lp.parent.mkdir(parents=True, exist_ok=True)
            image.save(ip)
            lp.write_text(f"{i%5} 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    dataset = output/"synthetic.yaml"
    dataset.write_text(yaml.safe_dump({"path": str((output/"synthetic").resolve()), **{s: f"images/{s}" for s in ("train", "val", "test")}, "names": {i: class_name(i) for i in range(5)}}), encoding="utf-8")
    device = "0" if gpu else "cpu"
    train = {**config["train"], "epochs": 1, "batch": 2, "imgsz": config["train"]["imgsz"] if gpu else 96, "mosaic": 0., "close_mosaic": 0}
    if gpu:
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(weights))
    actual_batches = []
    model.add_callback("on_train_batch_end", lambda trainer: actual_batches.append(int(trainer.batch_size)))
    start = time.perf_counter()
    model.train(**train, data=str(dataset.resolve()), device=device, seed=42, project=str(output.resolve()), name="train", exist_ok=False, save=True)
    smoke_config = {**config, "train": train, "evaluation": {**config["evaluation"], "bootstrap_resamples": 20}}
    metrics = evaluate_checkpoint(output/"train/weights/best.pt", dataset, smoke_config, output, device)
    receipt = {"state": "SYNTHETIC_GPU_PROBE_PASSED" if gpu else "SYNTHETIC_CPU_SMOKE_PASSED", "environment": env,
               "config_id": stable_id(config), "weights_sha256": file_sha256(weights), "plan_id": plan["plan_id"],
               "observed_batches": actual_batches, "expected_batch": train["batch"], "test_count": metrics["overall"]["image_count"],
               "peak_memory_bytes": int(torch.cuda.max_memory_allocated()) if gpu else None,
               "seconds": time.perf_counter()-start, "scientific_training_executed": False,
               "limitations": "Synthetic images, one epoch; not a representative FLIR VRAM/time estimate; Stage A still mandatory."}
    if not actual_batches or set(actual_batches) != {train["batch"]}:
        raise ValueError("Ultralytics changed batch size during probe")
    write_json(output/"probe.json", receipt)
    return receipt


def freeze_runtime(plan_directory: Path, probe: Path, output: Path) -> dict:
    configure_runtime()
    plan = verify_plan(plan_directory)
    config = copy.deepcopy(plan["identity"]["config"])
    env = environment()
    receipt = read_json(probe)
    if receipt.get("state") != "BATCH_PROBE_PASSED":
        raise ValueError("A successful hardware/batch probe is required before real training")
    weights = prepare_weights(output)
    if receipt["config_id"] != stable_id(config) or receipt["plan_id"] != plan["plan_id"] or receipt["weights_sha256"] != file_sha256(weights) or environment_signature(receipt["environment"]) != environment_signature(env):
        raise ValueError("Probe does not match the current frozen inputs/runtime")
    if env["packages"]["ultralytics"] != config["ultralytics_version"] or env["packages"]["torch"].split("+")[0] != config["torch_version"]:
        raise ValueError("Installed detector version differs from protocol")
    config["train"]["batch"] = receipt["batch"]
    model_config = {"protocol_config": config, "device": receipt["device"], "weights_sha256": file_sha256(weights),
                    "runtime_packages": env["packages"], "defaults_sha256": env["ultralytics_defaults_sha256"],
                    "tf32": False, "checkpoint_selection": "validation_only", "test_tuning": False}
    frozen = {"plan_id": plan["plan_id"], "model_config": model_config, "model_config_id": stable_id(model_config), "environment": env,
              "probe_sha256": file_sha256(probe), "provenance": provenance()}
    path = plan_directory/"runtime_freeze.json"
    if path.exists():
        existing = read_json(path)
        if existing["model_config"] != model_config or existing["plan_id"] != plan["plan_id"]:
            raise ValueError("Runtime freeze already exists with different scientific inputs")
        return existing
    write_json(path, frozen)
    return frozen


def parse_training_history(path: Path, expected_epochs: int) -> dict:
    history = pd.read_csv(path)
    history.columns = history.columns.str.strip()
    if len(history) != expected_epochs or history.epoch.tolist() != list(range(1, expected_epochs+1)):
        raise ValueError("Training did not complete the frozen epoch budget")
    column = "metrics/mAP50-95(B)"
    if not np.isfinite(history.select_dtypes(include="number").to_numpy()).all():
        raise ValueError("Non-finite training history")
    if column not in history:
        raise ValueError("Missing validation checkpoint-selection metric")
    # Ultralytics 8.3.203 uses mAP50-95 fitness and overwrites best on ties.
    best = history.loc[history[column] == history[column].max(), "epoch"].iloc[-1]
    return {"epochs_completed": len(history), "best_epoch_from_rounded_csv": int(best), "validation_map50_95": float(history[column].max())}


def batch_probe(plan_directory: Path, output: Path) -> dict:
    """Measure forward/backward/optimizer memory at final resolution, without FLIR."""
    configure_runtime()
    import gc

    import psutil
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds

    configure_runtime()
    plan = verify_plan(plan_directory)
    config = plan["identity"]["config"]
    env = environment()
    device = "0" if env["cuda_available"] else "cpu"
    candidates = config["compute"]["cuda_batch_candidates" if env["cuda_available"] else "cpu_batch_candidates"]
    weights = prepare_weights(output)
    trials = []
    for batch in candidates:
        init_seeds(42, deterministic=True)
        configure_runtime()
        started = time.perf_counter()
        model = tensor = loss = optimizer = None
        try:
            pretrained = YOLO(str(weights)).model
            model = DetectionModel(pretrained.yaml, nc=5, verbose=False)
            model.load(pretrained, verbose=False)
            del pretrained
            target_device = "cuda:0" if env["cuda_available"] else "cpu"
            model = model.to(target_device).train()
            model.args = type("LossArgs", (), config["train"])()
            optimizer = torch.optim.SGD(model.parameters(), lr=config["train"]["lr0"], momentum=config["train"]["momentum"])
            if env["cuda_available"]:
                torch.cuda.reset_peak_memory_stats()
            n = config["train"]["imgsz"]
            tensor = torch.rand(batch, 3, n, n, device=target_device)
            payload = {"img": tensor, "batch_idx": torch.arange(batch, device=target_device).repeat_interleave(5),
                       "cls": torch.arange(5, device=target_device).repeat(batch).reshape(-1, 1),
                       "bboxes": torch.tensor([[.2+c*.1, .5, .08, .08] for c in range(5)]*batch, device=target_device)}
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = model(payload)
                loss.sum().backward()
                optimizer.step()
            if env["cuda_available"]:
                torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated()) if env["cuda_available"] else psutil.Process().memory_info().rss
            safe = peak < .7*env["gpu_free_bytes"] if env["cuda_available"] else psutil.virtual_memory().available > 2*1024**3
            trials.append({"batch": batch, "seconds": time.perf_counter()-started, "peak_or_observed_process_bytes": peak, "safe": bool(safe)})
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            trials.append({"batch": batch, "safe": False, "error": "out_of_memory"})
            break
        finally:
            del model, tensor, loss, optimizer
            gc.collect()
            if env["cuda_available"]:
                torch.cuda.empty_cache()
    safe_batches = [t["batch"] for t in trials if t["safe"]]
    if not safe_batches:
        raise ValueError("No batch candidate passed the hardware probe")
    receipt = {"state": "BATCH_PROBE_PASSED", "plan_id": plan["plan_id"], "config_id": stable_id(config),
               "device": device, "batch": max(safe_batches), "environment": env, "trials": trials,
               "weights_sha256": file_sha256(weights), "scientific_training_executed": False,
               "limitation": "Synthetic five-box images; actual loader, augmentations and crowded FLIR scenes are checked in the pilot. CPU RSS is observed, not a sampled peak."}
    path = output/"batch_probe.json"
    if (plan_directory/"runtime_freeze.json").exists():
        raise ValueError("Runtime is already frozen; do not replace its batch probe")
    write_json(path, receipt)
    return receipt


def pilot_subset(records: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Deterministic within-partition coverage for an infrastructure pilot, not an evaluation sample."""
    order = records.sort_values("frame_id").sample(frac=1, random_state=seed)
    present = {}
    for row in order.itertuples():
        image = Path(row.image_path)
        text = (image.parent.parent/"labels"/f"{image.stem}.txt").read_text(encoding="utf-8-sig")
        present[row.frame_id] = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
    chosen = []
    for c in sorted(range(5), key=lambda c: sum(c in v for v in present.values())):
        if any(c in present[f] for f in chosen):
            continue
        possible = [f for f in order.frame_id if c in present[f]]
        if possible:
            chosen.append(possible[0])
    for f in order.frame_id:
        if len(chosen) >= limit:
            break
        if f not in chosen:
            chosen.append(f)
    if len(chosen) > limit:
        raise ValueError("Pilot limit cannot cover present classes")
    return records.set_index("frame_id").loc[chosen].reset_index().sort_values("frame_id")


def pilot_metadata(directory: Path, receipt: dict, freeze: dict) -> dict:
    """Derive the standard artifact schema without rewriting the execution receipt."""
    cohort = pd.read_parquet(directory/"pilot_records.parquet")
    model_config = {**freeze["model_config"], "protocol_config": receipt["pilot_configuration"],
                    "cohort_id": stable_id({"records": cohort[["frame_id", "new_split"]].sort_values("frame_id").to_dict("records")}),
                    "purpose": "small_infrastructure_pilot"}
    run_id = detector_run_id(receipt["split_space_id"], model_config, receipt["training_seed"])
    summary = {k: receipt[k] for k in ("epochs_completed", "best_epoch", "training_seconds", "batch", "image_size", "initial_weights_sha256")}
    summary["effective_arguments"] = yaml.safe_load((directory/"train/args.yaml").read_text(encoding="utf-8"))
    summary["source_execution_receipt_sha256"] = file_sha256(directory/"pilot.json")
    summary["scientific_result"] = False
    write_json(directory/"training_summary.json", summary)
    metadata = {**receipt["provenance"], "detector_run_id": run_id, "dataset_id": receipt["dataset_id"],
                "identity": {"split_space_id": receipt["split_space_id"], "model_configuration": model_config, "training_seed": receipt["training_seed"]},
                "strategy": receipt["strategy"], "split_seed": 0, "detector_seed": receipt["training_seed"],
                "environment": receipt["environment"], "state": "SMALL_PILOT_VALIDATED", "scientific_result": False,
                "source_execution_receipt_sha256": file_sha256(directory/"pilot.json"),
                "output_sha256": {**receipt["output_sha256"], "training_summary.json": file_sha256(directory/"training_summary.json"), "pilot.json": file_sha256(directory/"pilot.json")}}
    write_json(directory/"metadata.json", metadata)
    return metadata


def small_pilot(plan_directory: Path, output: Path) -> dict:
    """Tiny real-data operational pilot. Never promoted into Stage A/B results."""
    configure_runtime()
    from ultralytics import YOLO

    configure_runtime()
    plan = verify_plan(plan_directory)
    freeze = read_json(plan_directory/"runtime_freeze.json")
    config = freeze["model_config"]["protocol_config"]
    env = environment()
    if environment_signature(env) != environment_signature(freeze["environment"]):
        raise ValueError("Pilot runtime differs from its freeze")
    weights = prepare_weights(output)
    if file_sha256(weights) != freeze["model_config"]["weights_sha256"]:
        raise ValueError("Pilot pretrained initialization mismatch")
    device = freeze["model_config"]["device"]
    summaries = []
    for split in plan["identity"]["splits"]:
        if split["split_seed"] != 0:
            continue
        directory = output/"small_pilot"/split["strategy"]
        receipt_path = directory/"pilot.json"
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["model_config_id"] != freeze["model_config_id"] or receipt["split_space_id"] != split["split_space_id"]:
                raise ValueError("Small-pilot resume identity mismatch")
            if any(file_sha256(directory/n) != h for n, h in receipt["output_sha256"].items()):
                raise ValueError("Small-pilot output changed")
            pilot_metadata(directory, receipt, freeze)
            summaries.append(receipt)
            continue
        if (directory/"train").exists():
            raise ValueError("Partial small pilot preserved; inspect before rerunning")
        view = output/"views"/split["split_space_id"]
        materialization = verify_view(view)
        if materialization["split_metadata_sha256"] != split["split_metadata_sha256"]:
            raise ValueError("Small pilot view differs from selected source")
        records = pd.read_parquet(view/"records.parquet")
        directory.mkdir(parents=True, exist_ok=True)
        subsets = []
        for partition in ("train", "val", "test"):
            limit = config["compute"][f"small_pilot_{partition}_images"]
            subset = pilot_subset(records.loc[records.new_split == partition], limit, 42)
            subsets.append(subset)
            (directory/f"{partition}.txt").write_text("\n".join(subset.image_path)+"\n", encoding="utf-8")
        cohort = pd.concat(subsets, ignore_index=True)
        cohort.to_parquet(directory/"pilot_records.parquet", index=False)
        dataset = directory/"dataset.yaml"
        dataset.write_text(yaml.safe_dump({"path": str(directory.resolve()), **{s: str((directory/f"{s}.txt").resolve()) for s in ("train", "val", "test")}, "names": {i: class_name(i) for i in range(5)}}), encoding="utf-8")
        train = {**config["train"], "epochs": config["compute"]["small_pilot_epochs"], "close_mosaic": 0}
        pilot_config = {**config, "train": train}
        model = YOLO(str(weights))
        batches, epoch_times = [], []
        best = {"fitness": None, "epoch": None}

        def track(trainer, batches=batches, epoch_times=epoch_times, best=best):
            batches.append(int(trainer.batch_size))
            epoch_times.append(float(trainer.epoch_time))
            if best["fitness"] is None or trainer.fitness >= best["fitness"]:
                best.update(fitness=float(trainer.fitness), epoch=int(trainer.epoch)+1)

        model.add_callback("on_fit_epoch_end", track)
        expected_counts = cohort.set_index("frame_id").num_objects.to_dict()
        model.add_callback("on_train_start", lambda trainer, expected_counts=expected_counts: check_loader_annotations(trainer, expected_counts))
        started = time.perf_counter()
        model.train(**train, data=str(dataset.resolve()), device=device, seed=42, split="val", project=str(directory.resolve()), name="train", exist_ok=False, save=True)
        seconds = time.perf_counter()-started
        history = parse_training_history(directory/"train/results.csv", train["epochs"])
        if set(batches) != {train["batch"]}:
            raise ValueError("Pilot batch changed; do not proceed with incomparable settings")
        evaluation_started = time.perf_counter()
        metrics = evaluate_checkpoint(directory/"train/weights/best.pt", dataset, pilot_config, directory, device)
        from flir_pipeline.detection.metrics import load_image_stats

        expected = cohort.loc[cohort.new_split == "test"]
        observed = load_image_stats(directory/"test_image_stats.npz")
        if sorted(s.frame_id for s in observed) != sorted(expected.frame_id) or metrics["overall"]["supported_classes"] != 5:
            raise ValueError("Pilot test image/class coverage failed")
        if any(len(s.target_class) != expected_counts[s.frame_id] for s in observed):
            raise ValueError("Pilot test annotation support differs from canonical labels")
        counts = cohort.groupby("new_split").size().to_dict()
        # Conservative extrapolation from complete pilot wall time, including startup;
        # val scales separately only in the later full pilot. This is an estimate.
        scale = split["record_counts"]["train"]/counts["train"]
        hours = seconds/train["epochs"]*config["train"]["epochs"]*scale/3600
        receipt = {"state": "SMALL_PILOT_VALIDATED", "strategy": split["strategy"], "split_space_id": split["split_space_id"],
                   "model_config_id": freeze["model_config_id"], "dataset_id": split["dataset_id"], "training_seed": 42,
                   "counts": counts, "environment": env, "device": device, "batch": train["batch"], "image_size": train["imgsz"],
                   "training_seconds": seconds, "evaluation_bootstrap_seconds": time.perf_counter()-evaluation_started,
                   "epoch_seconds": epoch_times, "estimated_full_training_hours": hours, "best_epoch": best["epoch"], **history,
                   "scientific_result": False, "stage_a_completed": False, "source_split_membership_preserved": True,
                   "test_tuning": False, "initial_weights_sha256": file_sha256(weights),
                   "pilot_configuration": pilot_config, "provenance": provenance(),
                   "output_sha256": {n: file_sha256(directory/n) for n in ("metrics.json", "metrics_per_class.parquet", "test_image_stats.npz", "bootstrap.json", "bootstrap_samples.parquet", "pilot_records.parquet", "dataset.yaml", "train.txt", "val.txt", "test.txt", "train/weights/best.pt", "train/args.yaml", "train/results.csv")}}
        write_json(receipt_path, receipt)
        pilot_metadata(directory, receipt, freeze)
        summaries.append(receipt)
        print(f"Small pilot validated: {split['strategy']}, {seconds:.1f}s training; scientific Stage A/B pending", flush=True)
    costs = {r["strategy"]: r["estimated_full_training_hours"] for r in summaries}
    steady = {r["strategy"]: r["epoch_seconds"][r["epochs_completed"]-1]*config["train"]["epochs"]*
              next(s["record_counts"]["train"] for s in plan["identity"]["splits"] if s["strategy"] == r["strategy"] and s["split_seed"] == 0)/r["counts"]["train"]/3600 for r in summaries}
    expected_runs = len(plan["identity"]["splits"])*len(config["training_seeds"])
    total = sum(costs[s["strategy"]]*len(config["training_seeds"]) for s in plan["identity"]["splits"])
    steady_total = sum(steady[s["strategy"]]*len(config["training_seeds"]) for s in plan["identity"]["splits"])
    result = {"state": "SMALL_PILOTS_VALIDATED_STAGE_B_NOT_STARTED", "pilot_count": len(summaries), "expected_full_runs": expected_runs,
              "estimated_training_hours_by_strategy": costs, "estimated_full_training_hours": total,
              "estimated_last_epoch_training_hours_by_strategy": steady, "estimated_last_epoch_full_training_hours": steady_total,
              "estimation": "Linear extrapolation of tiny pilot wall time by training-image count and epochs; includes startup, excludes final test/bootstrap; not a measured full-run duration.",
              "last_epoch_estimation": "Alternative extrapolation from the last measured training/validation epoch to reduce startup overhead; not a confidence interval or a measured full-run duration. Raw epoch_seconds also includes the upstream final-validation callback; only epoch index epochs_completed-1 is used.",
              "device": device, "automatic_stage_b": False, "reason": "CPU cost must be reviewed before final training; pilot outcomes are not scientific comparisons."}
    write_json(output/"compute_budget.json", result)
    return result


def run_one(split: dict, seed: int, plan: dict, freeze: dict, output: Path) -> Path:
    configure_runtime()
    env = environment()
    if environment_signature(env) != environment_signature(freeze["environment"]):
        raise ValueError("Execution environment differs from runtime freeze")
    config = freeze["model_config"]["protocol_config"]
    model_config = freeze["model_config"]
    run_id = detector_run_id(split["split_space_id"], model_config, seed)
    run = output/"runs"/run_id
    view = output/"views"/split["split_space_id"]
    materialization = verify_view(view)
    if materialization["split_metadata_sha256"] != split["split_metadata_sha256"] or materialization["counts"] != split["record_counts"]:
        raise ValueError("Dataset view differs from frozen split")
    weights = prepare_weights(output)
    if file_sha256(weights) != model_config["weights_sha256"]:
        raise ValueError("Pretrained initialization changed")
    identity = {"split_space_id": split["split_space_id"], "model_configuration": model_config, "training_seed": seed}
    run.mkdir(parents=True, exist_ok=True)
    state_path = run/"state.json"
    if state_path.exists():
        state = read_json(state_path)
        if state["identity"] != identity:
            raise ValueError("Resume identity mismatch")
        if state["state"] == "COMPLETE":
            verify_run(run, split, freeze)
            return run
    else:
        state = {"identity": identity, "state": "INITIALIZED", "training_seconds": 0., "best_epoch": None, "best_fitness": None, "resume_count": 0}
        write_json(state_path, state)
    train_dir = run/"train"
    summary_path = run/"training_summary.json"
    if not summary_path.exists():
        import torch
        from ultralytics import YOLO

        last = train_dir/"weights/last.pt"
        if train_dir.exists() and not last.exists():
            raise ValueError("Partial run has no resumable last.pt; preserve it for inspection")
        model = YOLO(str(last if last.exists() else weights))
        started = time.perf_counter()
        prior_seconds = float(state["training_seconds"])
        torch.cuda.reset_peak_memory_stats()

        def control(trainer):
            if int(trainer.batch_size) != config["train"]["batch"] or trainer.args.split != "val":
                raise ValueError("Batch changed or validation uses the wrong partition")

        def checkpoint(trainer):
            control(trainer)
            fitness = float(trainer.fitness)
            if state["best_fitness"] is None or fitness >= state["best_fitness"]:
                state.update(best_epoch=int(trainer.epoch)+1, best_fitness=fitness)
            state.update(state="TRAINING", training_seconds=prior_seconds+time.perf_counter()-started,
                         peak_memory_bytes=int(torch.cuda.max_memory_allocated()), epoch=int(trainer.epoch)+1)
            write_json(state_path, state)

        model.add_callback("on_train_batch_end", control)
        model.add_callback("on_fit_epoch_end", checkpoint)
        counts = pd.read_parquet(view/"records.parquet").set_index("frame_id").num_objects.to_dict()
        model.add_callback("on_train_start", lambda trainer: check_loader_annotations(trainer, counts))
        if last.exists():
            state["resume_count"] += 1
            # Resume only this run; cross-strategy initialization is never allowed.
            if (train_dir/"results.csv").exists() and len(pd.read_csv(train_dir/"results.csv")) == config["train"]["epochs"]:
                pass  # Training finished before a process interruption; evaluate preserved best.pt.
            else:
                model.train(resume=True, device="0")
        else:
            model.train(**config["train"], data=str((view/"dataset.yaml").resolve()), device="0", seed=seed, split="val",
                        project=str(run.resolve()), name="train", exist_ok=False, save=True)
        history = parse_training_history(train_dir/"results.csv", config["train"]["epochs"])
        actual = yaml.safe_load((train_dir/"args.yaml").read_text(encoding="utf-8"))
        if any(actual.get(k) != v for k, v in config["train"].items()) or actual["seed"] != seed:
            raise ValueError("Effective training arguments changed")
        summary = {**history, "best_epoch": state["best_epoch"], "training_seconds": prior_seconds+time.perf_counter()-started,
                   "peak_memory_bytes": state.get("peak_memory_bytes"), "resume_count": state["resume_count"],
                   "initial_weights_sha256": file_sha256(weights), "effective_arguments": actual}
        write_json(summary_path, summary)
        state["state"] = "TRAINED"
        write_json(state_path, state)
    best = train_dir/"weights/best.pt"
    metrics = evaluate_checkpoint(best, view/"dataset.yaml", config, run)
    expected = pd.read_parquet(view/"records.parquet").query("new_split == 'test'")
    from flir_pipeline.detection.metrics import load_image_stats

    images = load_image_stats(run/"test_image_stats.npz")
    if sorted(s.frame_id for s in images) != sorted(expected.frame_id) or metrics["overall"]["image_count"] != split["record_counts"]["test"]:
        raise ValueError("Test image coverage differs from frozen membership")
    expected_counts = expected.set_index("frame_id").num_objects.to_dict()
    if any(len(s.target_class) != expected_counts[s.frame_id] for s in images):
        raise ValueError("Test annotation support changed")
    if metrics["overall"]["supported_classes"] != 5:
        raise ValueError("Test evaluation does not cover all five canonical classes")
    files = ["metrics.json", "metrics_per_class.parquet", "bootstrap.json", "bootstrap_samples.parquet", "test_image_stats.npz", "training_summary.json", "train/weights/best.pt", "train/results.csv", "train/args.yaml"]
    meta = {**provenance(), "detector_run_id": run_id, "identity": identity, "plan_id": plan["plan_id"],
            "strategy": split["strategy"], "split_seed": split["split_seed"], "detector_seed": seed, "dataset_id": split["dataset_id"],
            "environment": env, "context": split["context"], "materialization_sha256": file_sha256(view/"materialization.json"),
            "state": "COMPLETE", "test_tuning": False, "output_sha256": {n: file_sha256(run/n) for n in files}}
    write_json(run/"metadata.json", meta)
    verify_run(run, split, freeze)
    state["state"] = "COMPLETE"
    write_json(state_path, state)
    return run


def verify_run(run: Path, split: dict, freeze: dict) -> dict:
    meta = read_json(run/"metadata.json")
    if stable_id(meta["identity"]) != meta["detector_run_id"] or meta["identity"]["model_configuration"] != freeze["model_config"]:
        raise ValueError("Detector run identity/configuration mismatch")
    if meta["identity"]["split_space_id"] != split["split_space_id"] or meta["test_tuning"]:
        raise ValueError("Detector split/control mismatch")
    required = {"metrics.json", "metrics_per_class.parquet", "bootstrap.json", "bootstrap_samples.parquet", "test_image_stats.npz", "training_summary.json", "train/weights/best.pt", "train/results.csv", "train/args.yaml"}
    if set(meta["output_sha256"]) != required or any(file_sha256(run/n) != h for n, h in meta["output_sha256"].items()):
        raise ValueError("Detector output checksum mismatch")
    metrics = read_json(run/"metrics.json")
    classes = pd.read_parquet(run/"metrics_per_class.parquet")
    if classes.class_id.tolist() != list(range(5)) or not (classes.support > 0).all() or metrics["overall"]["image_count"] != split["record_counts"]["test"]:
        raise ValueError("Detector class/test coverage mismatch")
    if classes.class_name.tolist() != [class_name(i) for i in range(5)]:
        raise ValueError("Detector class names mismatch")
    from flir_pipeline.detection.protocol import METRICS

    if not np.isfinite(classes[list(METRICS)].to_numpy()).all():
        raise ValueError("Unreadable detector metrics")
    summary = read_json(run/"training_summary.json")
    config = freeze["model_config"]["protocol_config"]
    if summary["epochs_completed"] != config["train"]["epochs"] or summary["initial_weights_sha256"] != freeze["model_config"]["weights_sha256"]:
        raise ValueError("Incomplete training or different initialization")
    if any(summary["effective_arguments"].get(k) != v for k, v in config["train"].items()):
        raise ValueError("Effective configuration differs from protocol")
    if summary["effective_arguments"]["seed"] != meta["detector_seed"] or metrics["test_threshold_optimized"] or metrics["checkpoint_selection"] != "validation_only":
        raise ValueError("Evaluation or seed controls changed")
    expected_support = {c["class_id"]: c["instances"] for c in split["class_counts"] if c["split"] == "test"}
    if classes.set_index("class_id").support.to_dict() != expected_support:
        raise ValueError("Test annotation/class support differs from frozen split")
    from flir_pipeline.detection.metrics import load_image_stats

    images = load_image_stats(run/"test_image_stats.npz")
    view = run.parent.parent/"views"/split["split_space_id"]
    if file_sha256(view/"materialization.json") != meta["materialization_sha256"]:
        raise ValueError("Run dataset view receipt changed")
    receipt = read_json(view/"materialization.json")
    if receipt["split_metadata_sha256"] != split["split_metadata_sha256"] or file_sha256(view/"records.parquet") != receipt["files"]["records.parquet"]:
        raise ValueError("Run dataset source binding changed")
    expected = pd.read_parquet(view/"records.parquet").query("new_split == 'test'")
    if sorted(s.frame_id for s in images) != sorted(expected.frame_id):
        raise ValueError("Run evaluated a different test cohort")
    rebuilt, rebuilt_classes = evaluate(images)
    if rebuilt != metrics["overall"]:
        raise ValueError("Stored metrics differ from captured test statistics")
    pd.testing.assert_frame_equal(rebuilt_classes, classes)
    return meta


def fair_comparison(metas: list[dict], expected: pd.DataFrame) -> dict:
    errors = []
    expected_cells = set(zip(expected.split_space_id, expected.detector_seed, strict=True))
    actual_cells = [(m["identity"]["split_space_id"], m["detector_seed"]) for m in metas]
    if len(actual_cells) != len(set(actual_cells)) or set(actual_cells) != expected_cells:
        errors.append("missing_or_duplicate_experimental_cells")
    if metas and len({stable_id(m["identity"]["model_configuration"]) for m in metas}) != 1:
        errors.append("model_weights_or_hyperparameters_differ")
    if any(m["test_tuning"] or m["state"] != "COMPLETE" or m["identity"]["training_seed"] != m["detector_seed"] for m in metas):
        errors.append("test_tuning_incomplete_run_or_seed_mismatch")
    if metas and len({m["dataset_id"] for m in metas}) != 1:
        errors.append("dataset_identity_differs")
    return {"controlled": not errors, "errors": errors, "expected_runs": len(expected_cells), "completed_runs": len(actual_cells)}


def execute_matrix(plan_directory: Path, output: Path) -> dict:
    plan = verify_plan(plan_directory)
    freeze = read_json(plan_directory/"runtime_freeze.json")
    if freeze["plan_id"] != plan["plan_id"] or stable_id(freeze["model_config"]) != freeze["model_config_id"]:
        raise ValueError("Invalid execution freeze")
    if freeze["model_config"]["device"] == "cpu":
        raise ValueError("Automatic Stage B on CPU is disabled by the compute protocol; run pilot-small and review the cost estimate first")
    splits = plan["identity"]["splits"]
    matrix = experiment_matrix(splits, plan["identity"]["config"]["training_seeds"])
    lookup = {s["split_space_id"]: s for s in splits}
    pilots = []
    for cell in matrix.loc[matrix.pilot].itertuples():
        split = lookup[cell.split_space_id]
        directory = run_one(split, cell.detector_seed, plan, freeze, output)
        pilots.append(verify_run(directory, split, freeze))
    gate = fair_comparison(pilots, matrix.loc[matrix.pilot])
    write_json(plan_directory/"pilot_validation.json", gate)
    if not gate["controlled"]:
        raise ValueError("Stage A gate failed; Stage B cannot start")
    runs = []
    for cell in matrix.itertuples():
        split = lookup[cell.split_space_id]
        directory = run_one(split, cell.detector_seed, plan, freeze, output)
        runs.append(verify_run(directory, split, freeze))
        print(f"Verified {len(runs)}/{len(matrix)} {cell.strategy} split={cell.split_seed} detector={cell.detector_seed}", flush=True)
    fairness = fair_comparison(runs, matrix)
    write_json(plan_directory/"final_validation.json", fairness)
    return fairness
