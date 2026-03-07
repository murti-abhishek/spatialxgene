"""Tests for spatialxgene.cli — CLI entry points."""

from click.testing import CliRunner
from spatialxgene.cli import main


def test_main_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "spatialxgene" in result.output


def test_launch_help():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--help"])
    assert result.exit_code == 0
    assert "H5AD_FILE" in result.output
    assert "--host" in result.output
    assert "--port" in result.output
    assert "--subsample" in result.output
    assert "--skip-columns" in result.output


def test_launch_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "nonexistent.h5ad"])
    assert result.exit_code != 0


def test_launch_invalid_subsample():
    runner = CliRunner()
    result = runner.invoke(main, ["launch", "--subsample", "not_a_number", "dummy.h5ad"])
    assert result.exit_code != 0
