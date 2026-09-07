from pathlib import Path

from typer.testing import CliRunner

from flir_pipeline import __version__
from flir_pipeline.cli import app
from flir_pipeline.config import PipelineConfig

runner = CliRunner()


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Reproducible FLIR" in result.stdout


def test_base_config_loads() -> None:
    config = PipelineConfig.model_validate({"artifacts_root": "artifacts"})
    assert config.artifacts_root == Path("artifacts")
