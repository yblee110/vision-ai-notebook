"""Distribution must retain setup assets while excluding local state and old code."""
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest


spec = importlib.util.spec_from_file_location("student_distribution", Path(__file__).parents[1] / "scripts/package.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def fixture_root(tmp_path):
    root = tmp_path / "course"
    for name in package.REQUIRED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    return root


def test_zip_keeps_required_files_but_not_state_or_legacy(tmp_path):
    root = fixture_root(tmp_path)
    excluded = [
        ".local-state/colab-cli/token.json", ".venv/bin/python", ".cache/download.bin",
        "runs/gpu/report.json", "results/gpu/report.json", "custom_images/test/cup.jpg",
        "vision_lab/model.py", "pyproject.toml", "presentation/output/old.pptx",
        "validation/old/report.json", "validation/direct/gpu/finetuned_checkpoint.npz",
        "validation/direct/gpu/raw.log", "validation/direct/gpu/session.json",
        ".agents/skills/vision-colab/token.yaml", "scripts/credentials.py",
        "notebooks/02_gpu_finetuning_output.ipynb", "scripts/__pycache__/cached.py",
        "hf_colab_gpu/models/deit-tiny/pytorch_model.bin",
        "hf_colab_gpu/models/deit-tiny/config.json", "hf_colab_gpu/data/train.npz",
        "hf_colab_gpu/.cache/huggingface/token", "hf_colab_gpu/results/report.json",
        "hf_colab_gpu/notebooks/01_gpu_inference_output.ipynb",
        "hf_colab_gpu/validation/gpu/finetuned_model.safetensors",
        "hf_colab_gpu/validation/gpu/model.zip", "hf_colab_gpu/validation/gpu/token.json",
        "hf_colab_gpu/validation/gpu/finetuned_model/config.json",
        "hf_colab_gpu/validation/gpu/finetuned_model/report.json",
        "hf_colab_gpu/validation/gpu/raw.log", "hf_colab_gpu/credentials.md",
    ]
    for name in excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private or obsolete")
    evidence = root / "validation/direct/gpu/report.json"
    evidence.write_text('{"status": "completed"}')
    hf_evidence = root / "hf_colab_gpu/validation/gpu/report.json"
    hf_evidence.write_text('{"status": "completed", "framework": "pytorch"}')
    hf_executed = root / "hf_colab_gpu/validation/gpu/01_gpu_inference_output.ipynb"
    hf_executed.write_text('{"cells": []}')
    (root / "hf_colab_gpu/build_notebooks.py").write_text("# Standalone notebook builder")
    artifact = tmp_path / "students.zip"
    result = package.build_distribution(artifact, root)
    with ZipFile(artifact) as archive:
        names = {name.removeprefix("vision_ai_notbook/") for name in archive.namelist()}
    assert package.REQUIRED <= names
    assert not names.intersection(excluded)
    assert "validation/direct/gpu/report.json" in names
    assert "hf_colab_gpu/validation/gpu/report.json" in names
    assert "hf_colab_gpu/validation/gpu/01_gpu_inference_output.ipynb" in names
    assert "hf_colab_gpu/build_notebooks.py" in names
    assert result["files"] == len(names)


def test_missing_notebook_aborts_distribution(tmp_path):
    root = fixture_root(tmp_path)
    (root / "notebooks/03_tpu_and_compare.ipynb").unlink()
    with pytest.raises(FileNotFoundError, match="03_tpu_and_compare.ipynb"):
        package.build_distribution(tmp_path / "students.zip", root)


def test_missing_basic_hf_notebook_aborts_distribution(tmp_path):
    root = fixture_root(tmp_path)
    (root / "hf_colab_gpu/notebooks/01_gpu_inference.ipynb").unlink()
    with pytest.raises(FileNotFoundError, match="01_gpu_inference.ipynb"):
        package.build_distribution(tmp_path / "students.zip", root)


def test_symlinked_source_cannot_add_external_files(tmp_path):
    root = fixture_root(tmp_path)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "secret.py").write_text("do not distribute")
    (root / "scripts/linked").symlink_to(outside, target_is_directory=True)
    (root / "scripts/other.py").symlink_to(outside / "secret.py")
    files = {path.relative_to(root).as_posix() for path in package.collect_files(root)}
    assert "scripts/other.py" not in files
    assert "scripts/linked/secret.py" not in files
