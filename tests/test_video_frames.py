"""Synthetic subprocess contracts; no FFmpeg installation or source videos needed."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from flir_pipeline.cli import app
from flir_pipeline.data import video_frames as vf


@pytest.fixture
def tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        calls=[],
        frames=3,
        fail=None,
        malformed=False,
        gap=False,
        stream={
            "codec_name": "h264",
            "width": 16,
            "height": 12,
            "r_frame_rate": "30/1",
            "avg_frame_rate": "30000/1001",
            "duration": "2.5",
            "nb_frames": "75",
        },
    )

    def run(arguments, **kwargs):
        assert isinstance(arguments, list)
        assert kwargs["shell"] is False
        assert kwargs["check"] is True
        assert kwargs["stdin"] == subprocess.DEVNULL
        state.calls.append(arguments)
        name = Path(arguments[0]).name
        if "-version" in arguments:
            return subprocess.CompletedProcess(
                arguments,
                0,
                f"{name} version synthetic-1\nconfiguration: /private/build/path",
                "",
            )
        if name == state.fail:
            raise subprocess.CalledProcessError(
                7, arguments, stderr="synthetic decoder error"
            )
        if name == "ffprobe":
            payload = {"streams": [state.stream], "format": {"duration": "3.0"}}
            return subprocess.CompletedProcess(
                arguments, 0, "not JSON" if state.malformed else json.dumps(payload), ""
            )
        assert name == "ffmpeg"
        for index in range(state.frames):
            actual = index + 1 if state.gap else index
            destination = kwargs["cwd"] / (arguments[-1] % actual)
            Image.new("RGB", (16, 12), (index, 20, 40)).save(destination)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(
        vf.shutil, "which", lambda name: str(tmp_path / Path(name).name)
    )
    monkeypatch.setattr(vf.subprocess, "run", run)
    return state


def source(root: Path, name: str = "nested/video.mp4") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"synthetic video placeholder")
    return path


def snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30/1", 30.0), ("30000/1001", 29.97002997002997), ("25", 25.0)],
)
def test_parse_frame_rate(value: str, expected: float) -> None:
    assert vf.parse_frame_rate(value) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value", [None, "0/0", "0/1", "-30/1", "N/A", "nan", "inf", "30/0", "oops"]
)
def test_invalid_rate_is_unknown(value: str | None) -> None:
    assert vf.parse_frame_rate(value) is None


def test_discovery_order_extensions_and_ids(tmp_path: Path) -> None:
    root = tmp_path / "source"
    for name in [
        "z.mkv",
        "folder/same.mov",
        "same.mp4",
        "b.AVI",
        "A.MP4",
        "same.mov",
        "ignore.webm",
        "image.jpg",
        "original.zip",
    ]:
        source(root, name)
    (root / "directory.mp4").mkdir()
    paths = [path.relative_to(root).as_posix() for path in vf.discover_videos(root)]
    assert paths == [
        "A.MP4",
        "b.AVI",
        "folder/same.mov",
        "same.mov",
        "same.mp4",
        "z.mkv",
    ]
    assert len({vf.video_id_for(path) for path in paths}) == len(paths)
    relocated = tmp_path / "relocated"
    for name in paths:
        source(relocated, name)
    assert [
        vf.video_id_for(path.relative_to(relocated).as_posix())
        for path in vf.discover_videos(relocated)
    ] == [vf.video_id_for(path) for path in paths]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_reject_invalid_sample_fps(tmp_path: Path, fps: float) -> None:
    with pytest.raises(ValueError, match="sample_fps"):
        vf.extract_video_frames(
            tmp_path / "source", tmp_path / "output", sample_fps=fps
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("quality", [0, 32, 1.5])
def test_reject_invalid_jpeg_quality(tmp_path: Path, quality) -> None:
    with pytest.raises(ValueError, match="jpeg_quality"):
        vf.extract_video_frames(
            tmp_path / "source", tmp_path / "output", jpeg_quality=quality
        )


def test_metadata_portability_grid_and_reproducibility(tmp_path: Path, tools) -> None:
    root, output = tmp_path / "source", tmp_path / "output 100%"
    video = source(root)
    source_before = snapshot(root)
    summary = vf.extract_video_frames(root, output, sample_fps=2)
    table = pd.read_parquet(output / "frames.parquet")
    assert list(table.columns) == list(vf.FRAME_DTYPES)
    assert table.source_video.tolist() == ["nested/video.mp4"] * 3
    assert table.sample_index.tolist() == [0, 1, 2]
    assert table.timestamp_seconds.tolist() == [0, 0.5, 1]
    assert table.source_frame_index_estimate.tolist() == [0, 15, 30]
    assert table.source_fps.tolist() == pytest.approx([30000 / 1001] * 3)
    assert table.source_width.tolist() == [16] * 3
    assert table.source_height.tolist() == [12] * 3
    assert table.source_duration_seconds.tolist() == [2.5] * 3
    assert table.source_nb_frames.tolist() == [75] * 3
    assert table.codec_name.tolist() == ["h264"] * 3
    assert table.sample_fps.tolist() == [2] * 3
    assert summary["processed_videos"] == 1 and summary["total_frames"] == 3
    assert summary["read_only_source"] is True
    assert summary["videos"][0]["source_fps_field"] == "avg_frame_rate"
    assert summary["videos"][0]["source_sha256"] == vf.sha256_file(video)
    assert json.loads((output / "summary.json").read_text()) == summary
    serialized = json.dumps(summary) + table.to_json()
    assert str(tmp_path) not in serialized and "/private/build/path" not in serialized
    for relative in table.image_path:
        assert "\\" not in relative and not Path(relative).is_absolute()
        assert (output / relative).is_file()
    calls = [
        args
        for args in tools.calls
        if Path(args[0]).name == "ffmpeg" and "-version" not in args
    ]
    assert len(calls) == 1
    command = calls[0]
    assert (
        command[command.index("-vf") + 1]
        == "setpts=PTS-STARTPTS,fps=fps=2.0:start_time=0:round=near:eof_action=pass"
    )
    assert command[command.index("-map") + 1] == "0:v:0"
    assert command[command.index("-start_number") + 1] == "0"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-q:v") + 1] == "2"
    assert "-noautorotate" in command and "-n" in command
    before = snapshot(output)
    assert (
        vf.extract_video_frames(root, output, sample_fps=2, overwrite=True) == summary
    )
    assert snapshot(output) == before
    assert snapshot(root) == source_before


@pytest.mark.parametrize(
    ("avg", "nominal", "expected", "field"),
    [
        ("0/0", "30/1", 30, "r_frame_rate"),
        (None, "30000/1001", 30000 / 1001, "r_frame_rate"),
        ("N/A", "0/0", None, None),
    ],
)
def test_probe_fallback_unknown_and_nullable_metadata(
    tmp_path: Path, tools, avg, nominal, expected, field
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    tools.stream.update(
        avg_frame_rate=avg, r_frame_rate=nominal, duration="N/A", nb_frames="N/A"
    )
    summary = vf.extract_video_frames(root, output)
    info = summary["videos"][0]
    assert info["source_fps"] == expected
    assert info["source_fps_field"] == field
    assert info["source_duration_seconds"] == 3.0
    assert info["duration_field"] == "format.duration"
    table = pd.read_parquet(output / "frames.parquet")
    assert table.source_nb_frames.isna().all()
    assert str(table.source_frame_index_estimate.dtype) == "Int64"
    if expected is None:
        assert table.source_fps.isna().all()
        assert table.source_frame_index_estimate.isna().all()


def test_estimated_index_rounds_half_up(tmp_path: Path, tools) -> None:
    source(tmp_path / "source")
    tools.stream["avg_frame_rate"] = "1/1"
    vf.extract_video_frames(tmp_path / "source", tmp_path / "output", sample_fps=2)
    assert pd.read_parquet(
        tmp_path / "output/frames.parquet"
    ).source_frame_index_estimate.tolist() == [0, 1, 1]


def test_no_overwrite_fails_before_any_tool_or_modification(
    tmp_path: Path, tools
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root, "a.mp4")
    vf.extract_video_frames(root, output)
    source(root, "b.mp4")
    before, calls = snapshot(output), len(tools.calls)
    with pytest.raises(vf.VideoFramesError, match="--overwrite"):
        vf.extract_video_frames(root, output)
    assert snapshot(output) == before and len(tools.calls) == calls


def test_safe_overwrite_preserves_unmanaged_files_and_removes_stale_frames(
    tmp_path: Path, tools
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    summary = vf.extract_video_frames(root, output)
    identity = summary["videos"][0]["video_id"]
    (output / "notes.txt").write_text("keep")
    (output / identity / "manual.jpg").write_bytes(b"user file")
    tools.frames = 1
    result = vf.extract_video_frames(root, output, overwrite=True)
    assert result["total_frames"] == 1
    assert not (output / identity / "frame_000002.jpg").exists()
    assert not (output / identity / "frame_000001.jpg").exists()
    assert (output / "notes.txt").read_text() == "keep"
    assert (output / identity / "manual.jpg").read_bytes() == b"user file"
    assert len(pd.read_parquet(output / "frames.parquet")) == 1


@pytest.mark.parametrize("overwrite", [False, True])
def test_unmanaged_collision_across_videos_is_preflighted(
    tmp_path: Path, tools, overwrite: bool
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root, "a.mp4")
    source(root, "z.mp4")
    directory = output / vf.video_id_for("z.mp4")
    directory.mkdir(parents=True)
    (directory / "frame_000000.jpg").write_bytes(b"user")
    before = snapshot(output)
    with pytest.raises(vf.VideoFramesError, match="Unmanaged frame collision"):
        vf.extract_video_frames(root, output, overwrite=overwrite)
    assert snapshot(output) == before and not tools.calls


@pytest.mark.parametrize("filename", ["summary.json", "frames.parquet"])
def test_overwrite_refuses_unrecognized_metadata(
    tmp_path: Path, tools, filename: str
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    output.mkdir()
    (output / filename).write_text("user file")
    before = snapshot(output)
    with pytest.raises(vf.VideoFramesError, match="Cannot safely overwrite"):
        vf.extract_video_frames(root, output, overwrite=True)
    assert snapshot(output) == before and not tools.calls


@pytest.mark.parametrize("tool", ["ffprobe", "ffmpeg"])
@pytest.mark.parametrize("existing", [False, True])
def test_subprocess_errors_preserve_results(
    tmp_path: Path, tools, tool: str, existing: bool
) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    if existing:
        vf.extract_video_frames(root, output)
    before = snapshot(output)
    tools.fail = tool
    with pytest.raises(
        vf.VideoFramesError,
        match=f"{tool} for nested/video.mp4.*synthetic decoder error",
    ):
        vf.extract_video_frames(root, output, overwrite=existing)
    assert snapshot(output) == before
    assert not list(output.glob(".video-frames-*"))


def test_invalid_probe_json(tmp_path: Path, tools) -> None:
    source(tmp_path / "source")
    tools.malformed = True
    with pytest.raises(
        vf.VideoFramesError, match="Cannot inspect nested/video.mp4.*ffprobe JSON"
    ):
        vf.extract_video_frames(tmp_path / "source", tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["width", "codec_name"])
def test_missing_required_probe_field(tmp_path: Path, tools, field: str) -> None:
    source(tmp_path / "source")
    del tools.stream[field]
    with pytest.raises(vf.VideoFramesError, match="Cannot inspect"):
        vf.extract_video_frames(tmp_path / "source", tmp_path / "output")


@pytest.mark.parametrize("tool", ["ffmpeg", "ffprobe"])
def test_missing_executables_are_clear(
    tmp_path: Path, tools, monkeypatch, tool: str
) -> None:
    source(tmp_path / "source")
    monkeypatch.setattr(
        vf.shutil, "which", lambda name: None if name == tool else str(tmp_path / name)
    )
    with pytest.raises(vf.VideoFramesError, match=f"{tool} executable not found"):
        vf.extract_video_frames(tmp_path / "source", tmp_path / "output")
    assert not (tmp_path / "output").exists() and not tools.calls


@pytest.mark.parametrize("overlap", ["same", "output_inside", "source_inside"])
def test_source_and_output_roots_must_be_disjoint(
    tmp_path: Path, tools, overlap: str
) -> None:
    root = tmp_path / "source"
    source(root)
    output = {
        "same": root,
        "output_inside": root / "output",
        "source_inside": tmp_path,
    }[overlap]
    with pytest.raises(vf.VideoFramesError, match="disjoint"):
        vf.extract_video_frames(root, output)
    assert not tools.calls


@pytest.mark.parametrize("frames,gap", [(0, False), (2, True)])
def test_no_frames_or_gapped_sequence_is_not_published(
    tmp_path: Path, tools, frames: int, gap: bool
) -> None:
    source(tmp_path / "source")
    tools.frames, tools.gap = frames, gap
    with pytest.raises(vf.VideoFramesError, match="no frames or a non-contiguous"):
        vf.extract_video_frames(tmp_path / "source", tmp_path / "output")
    assert snapshot(tmp_path / "output") == {}


def test_tampered_receipt_cannot_delete_arbitrary_files(tmp_path: Path, tools) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    summary = vf.extract_video_frames(root, output)
    summary["videos"][0]["video_id"] = "../victim"
    (output / "summary.json").write_text(json.dumps(summary))
    before = snapshot(output)
    with pytest.raises(
        vf.VideoFramesError, match="invalid or duplicate video identity"
    ):
        vf.extract_video_frames(root, output, overwrite=True)
    assert snapshot(output) == before


def test_incomplete_publication_is_preserved(tmp_path: Path, tools) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    output.mkdir()
    (output / vf.PUBLICATION_MARKER).write_text("interrupted")
    with pytest.raises(vf.VideoFramesError, match="Incomplete publication"):
        vf.extract_video_frames(root, output, overwrite=True)
    assert (output / vf.PUBLICATION_MARKER).read_text() == "interrupted"
    assert not tools.calls


def test_cli_integration_and_failure(tmp_path: Path, tools) -> None:
    root, output = tmp_path / "source", tmp_path / "output"
    source(root)
    runner = CliRunner()
    help_result = runner.invoke(app, ["data", "extract-video-frames", "--help"])
    assert help_result.exit_code == 0
    assert "extract-video-frames" in help_result.stdout
    args = [
        "data",
        "extract-video-frames",
        "--videos-root",
        str(root),
        "--output-root",
        str(output),
    ]
    result = runner.invoke(app, args + ["--sample-fps", "2", "--jpeg-quality", "3"])
    assert result.exit_code == 0, result.output
    assert "1 videos, 3 frames" in result.output
    assert json.loads((output / "summary.json").read_text())["jpeg_quality"] == 3
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "--overwrite" in result.output
    result = runner.invoke(app, args + ["--overwrite"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, args + ["--sample-fps", "0"])
    assert result.exit_code == 1 and "sample_fps" in result.output
    tools.fail = "ffmpeg"
    result = runner.invoke(app, args + ["--overwrite"])
    assert result.exit_code == 1 and "synthetic decoder error" in result.output


def test_cli_requires_both_roots() -> None:
    result = CliRunner().invoke(app, ["data", "extract-video-frames"])
    assert result.exit_code != 0 and "Missing option" in result.output
