"""Read-only video sampling; grid times are not decoder or capture timestamps."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from flir_pipeline.data.inventory import VIDEO_EXTENSIONS
from flir_pipeline.utils.hashing import sha256_file

PRODUCER = "flir-pipeline.data.extract-video-frames"
SCHEMA_VERSION = 1
PUBLICATION_MARKER = ".video-frames-publishing"
FRAME_DTYPES = {
    "video_id": "string",
    "source_video": "string",
    "sample_index": "int64",
    "timestamp_seconds": "float64",
    "source_fps": "Float64",
    "source_frame_index_estimate": "Int64",
    "source_width": "int64",
    "source_height": "int64",
    "source_duration_seconds": "Float64",
    "source_nb_frames": "Int64",
    "codec_name": "string",
    "sample_fps": "float64",
    "image_path": "string",
}


class VideoFramesError(RuntimeError):
    """An inspection, extraction or safe-publication failure."""


@dataclass(frozen=True)
class VideoInfo:
    """Reported stream properties; unavailable optional facts stay null."""

    source_fps: float | None
    source_width: int
    source_height: int
    source_duration_seconds: float | None
    source_nb_frames: int | None
    codec_name: str
    avg_frame_rate: str | None
    r_frame_rate: str | None
    source_fps_field: str | None
    duration_field: str | None


def parse_frame_rate(value: str | None) -> float | None:
    """Parse a positive rational rate; 0/0, N/A and missing rates are unknown."""
    try:
        rate = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def _portable_source(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and not path.is_absolute()
        and not any(
            part in {".", ".."} or "\\" in part or ":" in part
            for part in value.split("/")
        )
        and path.as_posix() == value
    )


def video_id_for(source_video: str) -> str:
    """Namespace the relative filename, not a sequence or exact visual content."""
    return "video_" + hashlib.sha256(source_video.encode("utf-8")).hexdigest()[:16]


def discover_videos(videos_root: Path) -> list[Path]:
    """Recursively sort supported files by case-sensitive portable relative path."""
    root = videos_root.expanduser().resolve()
    if not root.is_dir():
        raise VideoFramesError(f"Videos root is not a directory: {videos_root}")
    videos = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    for path in videos:
        relative = path.relative_to(root).as_posix()
        if not path.resolve().is_relative_to(root) or not _portable_source(relative):
            raise VideoFramesError(
                f"Video path escapes the source root or is not portable: {relative}"
            )
    return videos


def _binary(value: str | Path, tool: str) -> str:
    executable = shutil.which(str(Path(value).expanduser()))
    if executable is None:
        raise VideoFramesError(
            f"{tool} executable not found: {value}. Use --{tool}-bin or install it on PATH."
        )
    return str(Path(executable).resolve())


def _run(arguments: list[str], context: str, *, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            arguments,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
            shell=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "No stderr supplied")[-6000:].strip()
        raise VideoFramesError(
            f"{context} failed (exit {error.returncode}): {detail}"
        ) from error
    except OSError as error:
        raise VideoFramesError(f"Cannot run {context}: {error}") from error
    return result.stdout


def _optional_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


def probe_video(video: Path, ffprobe_bin: str, source_video: str) -> VideoInfo:
    """Inspect only the first video stream, matching FFmpeg's explicit mapping."""
    output = _run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,duration,nb_frames:format=duration",
            "-of",
            "json",
            str(video),
        ],
        f"ffprobe for {source_video}",
    )
    try:
        payload = json.loads(output)
        stream = payload["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        codec = stream["codec_name"]
        if width <= 0 or height <= 0 or not isinstance(codec, str) or not codec:
            raise ValueError("missing codec or positive dimensions")
        avg, nominal = stream.get("avg_frame_rate"), stream.get("r_frame_rate")
        fps, field = parse_frame_rate(avg), "avg_frame_rate"
        if fps is None:
            fps, field = parse_frame_rate(nominal), "r_frame_rate"
        duration, duration_field = (
            _optional_duration(stream.get("duration")),
            "stream.duration",
        )
        if duration is None:
            duration = _optional_duration(payload.get("format", {}).get("duration"))
            duration_field = "format.duration"
        reported_count = str(stream.get("nb_frames", ""))
        count = int(reported_count) if re.fullmatch(r"[0-9]+", reported_count) else None
        return VideoInfo(
            fps,
            width,
            height,
            duration,
            count,
            codec,
            avg,
            nominal,
            field if fps is not None else None,
            duration_field if duration is not None else None,
        )
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise VideoFramesError(
            f"Cannot inspect {source_video}: invalid ffprobe JSON or video stream ({error})."
        ) from error


def _safe_output(root: Path, relative: str) -> Path:
    """Refuse redirected paths before any overwrite or removal."""
    path = root / relative
    resolved = path.resolve()
    if (
        path.is_symlink()
        or resolved != path.absolute()
        or not resolved.is_relative_to(root)
    ):
        raise VideoFramesError(f"Output path is redirected or unsafe: {relative}")
    if path.exists() and not path.is_file():
        raise VideoFramesError(f"Expected an output file: {relative}")
    return path


def _previous_frames(root: Path, overwrite: bool) -> set[str]:
    """Only a completed receipt can authorize replacing named generated files."""
    if (root / PUBLICATION_MARKER).exists() or (root / PUBLICATION_MARKER).is_symlink():
        raise VideoFramesError(
            "Incomplete publication found; preserve it for inspection and use a new output root."
        )
    summary_path = _safe_output(root, "summary.json")
    parquet_path = _safe_output(root, "frames.parquet")
    if not summary_path.exists() and not parquet_path.exists():
        return set()
    if not overwrite:
        raise VideoFramesError(
            "Results already exist; use --overwrite or a different output root."
        )
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary["producer"] != PRODUCER
            or summary["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unrecognized producer/schema")
        if sha256_file(parquet_path) != summary["frames_parquet_sha256"]:
            raise ValueError("frames.parquet checksum mismatch")
        owned: set[str] = set()
        ids: set[str] = set()
        for video in summary["videos"]:
            source, identity, count = (
                video["source_video"],
                video["video_id"],
                video["extracted_frames"],
            )
            if (
                not _portable_source(source)
                or identity != video_id_for(source)
                or identity in ids
            ):
                raise ValueError("invalid or duplicate video identity")
            if type(count) is not int or count < 0:
                raise ValueError("invalid extracted frame count")
            ids.add(identity)
            owned.update(f"{identity}/frame_{index:06d}.jpg" for index in range(count))
        frame_table = pd.read_parquet(parquet_path, columns=["image_path"])
        if len(frame_table) != len(owned) or set(frame_table.image_path) != owned:
            raise ValueError("receipt and frame table disagree")
        for relative in sorted(owned):
            if not _safe_output(root, relative).is_file():
                raise ValueError(f"missing managed frame: {relative}")
        return owned
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        raise VideoFramesError(
            f"Cannot safely overwrite unrecognized/incomplete results: {error}"
        ) from error


def _check_destinations(root: Path, identities: list[str], owned: set[str]) -> None:
    for identity in identities:
        directory = root / identity
        if directory.is_symlink() or directory.resolve() != directory.absolute():
            raise VideoFramesError(f"Video output directory is redirected: {identity}")
        if directory.exists():
            if not directory.is_dir():
                raise VideoFramesError(
                    f"Video output path is not a directory: {identity}"
                )
            for path in directory.iterdir():
                if (
                    re.fullmatch(r"frame_[0-9]{6,}\.jpg", path.name)
                    and f"{identity}/{path.name}" not in owned
                ):
                    raise VideoFramesError(
                        f"Unmanaged frame collision: {identity}/{path.name}"
                    )


def _extract_video(
    video: Path,
    source: str,
    identity: str,
    stage: Path,
    executable: str,
    sample_fps: float,
    jpeg_quality: int,
) -> list[str]:
    (stage / identity).mkdir()
    # Relative image2 pattern avoids interpreting '%' in the user's output root.
    _run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-noautorotate",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            f"setpts=PTS-STARTPTS,fps=fps={sample_fps}:start_time=0:round=near:eof_action=pass",
            "-fps_mode",
            "passthrough",
            "-c:v",
            "mjpeg",
            "-q:v",
            str(jpeg_quality),
            "-threads:v",
            "1",
            "-map_metadata",
            "-1",
            "-f",
            "image2",
            "-start_number",
            "0",
            f"{identity}/frame_%06d.jpg",
        ],
        f"ffmpeg for {source}",
        cwd=stage,
    )
    paths = list((stage / identity).iterdir())
    expected = [f"{identity}/frame_{index:06d}.jpg" for index in range(len(paths))]
    if not expected or {path.name for path in paths} != {
        Path(name).name for name in expected
    }:
        raise VideoFramesError(
            f"ffmpeg for {source} produced no frames or a non-contiguous image sequence."
        )
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise VideoFramesError(
            f"ffmpeg for {source} produced an empty/invalid output file."
        )
    return expected


def extract_video_frames(
    videos_root: Path,
    output_root: Path,
    *,
    sample_fps: float = 1.0,
    ffmpeg_bin: str | Path = "ffmpeg",
    ffprobe_bin: str | Path = "ffprobe",
    jpeg_quality: int = 2,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Stage the entire sample before publishing; never mutate source videos.

    Overwrite replaces only files owned by a verified prior receipt. Publication
    is single-writer and not a multi-file filesystem transaction; a marker blocks
    reuse after interruption during promotion. Decoder failures precede promotion.
    """
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("sample_fps must be finite and > 0")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 31:
        raise ValueError("jpeg_quality must be an integer between 1 and 31")
    sample_fps = float(sample_fps)
    source_root, root = (
        videos_root.expanduser().resolve(),
        output_root.expanduser().resolve(),
    )
    if root.is_relative_to(source_root) or source_root.is_relative_to(root):
        raise VideoFramesError(
            "videos-root and output-root must be disjoint (neither may contain the other)."
        )
    if root.exists() and not root.is_dir():
        raise VideoFramesError("output-root must be a directory")
    videos = discover_videos(source_root)
    if not videos:
        raise VideoFramesError("No supported videos found (.mp4, .mov, .avi, .mkv).")
    sources = [video.relative_to(source_root).as_posix() for video in videos]
    identities = [video_id_for(source) for source in sources]
    if len(set(identities)) != len(identities):
        raise VideoFramesError("Video ID collision; no files were written.")
    owned = _previous_frames(root, overwrite)
    _check_destinations(root, identities, owned)
    ffmpeg, ffprobe = _binary(ffmpeg_bin, "ffmpeg"), _binary(ffprobe_bin, "ffprobe")
    versions = {}
    for name, executable in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)):
        version = _run([executable, "-version"], f"{name} version").split()
        if len(version) < 3 or version[:2] != [name, "version"]:
            raise VideoFramesError(f"Cannot identify {name} version.")
        versions[name] = version[2]
    infos = [
        probe_video(video, ffprobe, source)
        for video, source in zip(videos, sources, strict=True)
    ]
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".video-frames-", dir=root) as temporary:
        stage = Path(temporary)
        rows, video_summaries = [], []
        for video, source, identity, info in zip(
            videos, sources, identities, infos, strict=True
        ):
            before = video.stat()
            source_hash = sha256_file(video)
            images = _extract_video(
                video, source, identity, stage, ffmpeg, sample_fps, jpeg_quality
            )
            after = video.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise VideoFramesError(f"Source changed while sampling: {source}")
            for index, relative in enumerate(images):
                timestamp = index / sample_fps
                # Nearest nominal index, ties upward. VFR/gaps invalidate exact indexing.
                estimate = (
                    math.floor(timestamp * info.source_fps + 0.5)
                    if info.source_fps is not None
                    else None
                )
                rows.append(
                    {
                        "video_id": identity,
                        "source_video": source,
                        "sample_index": index,
                        "timestamp_seconds": timestamp,
                        **{
                            key: value
                            for key, value in asdict(info).items()
                            if key in FRAME_DTYPES
                        },
                        "source_frame_index_estimate": estimate,
                        "sample_fps": sample_fps,
                        "image_path": relative,
                    }
                )
            video_summaries.append(
                {
                    "video_id": identity,
                    "source_video": source,
                    **asdict(info),
                    "source_sha256": source_hash,
                    "extracted_frames": len(images),
                }
            )
        frame_table = pd.DataFrame(rows, columns=list(FRAME_DTYPES)).astype(
            FRAME_DTYPES
        )
        frame_table.to_parquet(stage / "frames.parquet", index=False)
        summary = {
            "producer": PRODUCER,
            "schema_version": SCHEMA_VERSION,
            "processed_videos": len(videos),
            "total_frames": len(rows),
            "sample_fps": sample_fps,
            "jpeg_quality": jpeg_quality,
            "read_only_source": True,
            "tool_versions": versions,
            "python_version": platform.python_version(),
            "implementation_sha256": sha256_file(Path(__file__)),
            "timestamp_basis": "sample_index / sample_fps; first decoded video PTS rebased to zero",
            "source_frame_index_estimate_rule": "floor(timestamp_seconds * source_fps + 0.5); not a decoder index",
            "sampling_filter": f"setpts=PTS-STARTPTS,fps=fps={sample_fps}:start_time=0:round=near:eof_action=pass",
            "autorotate": False,
            "video_stream": "0:v:0",
            "threads": 1,
            "frames_parquet_sha256": sha256_file(stage / "frames.parquet"),
            "videos": video_summaries,
        }
        (stage / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        # Recheck every destination before promotion; never glob-delete a directory.
        new_frames = set(frame_table.image_path)
        unowned_frames = new_frames - owned
        for relative in sorted(owned | new_frames | {"frames.parquet", "summary.json"}):
            path = _safe_output(root, relative)
            if relative in unowned_frames and path.exists():
                raise VideoFramesError(f"Unmanaged frame collision: {relative}")
        marker = root / PUBLICATION_MARKER
        with marker.open("x", encoding="utf-8") as handle:
            handle.write(
                "Publication in progress; if interrupted, preserve for inspection.\n"
            )
        for relative in sorted(new_frames):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            (stage / relative).replace(destination)
        for relative in sorted(owned - new_frames):
            _safe_output(root, relative).unlink()
        (stage / "frames.parquet").replace(root / "frames.parquet")
        (stage / "summary.json").replace(root / "summary.json")
        marker.unlink()
    return summary
