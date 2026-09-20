"""Bundle standalone notebooks and public-library requirements for students.

This ZIP distributes files; it does not build or install a Python package.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {
    "scripts": {".py", ".sh"},
    "tests": {".py"},
    "notebook_sources": {".py"},
    "docs": {".md"},
    "requirements": {".txt"},
    ".devcontainer": {".json", ".sh"},
    ".vscode": {".json"},
    ".agents/skills": {".md", ".yaml", ".yml", ".py"},
}
NOTEBOOKS = (
    "00_setup_and_data.ipynb", "01_pretrained_inference.ipynb",
    "02_gpu_finetuning.ipynb", "03_tpu_and_compare.ipynb", "04_custom_images.ipynb",
)
HF_NOTEBOOKS = (
    "00_hf_download_and_data.ipynb", "01_gpu_inference.ipynb", "02_gpu_finetuning.ipynb",
)
HF_FILES = (
    "hf_colab_gpu/README.md", "hf_colab_gpu/requirements-torch.txt",
    "hf_colab_gpu/requirements-gpu.txt",
    *(f"hf_colab_gpu/notebooks/{name}" for name in HF_NOTEBOOKS),
)
FILES = (
    "README.md", "THIRD_PARTY.md", "pytest.ini", ".vision-lab-root", ".gitignore",
    ".dockerignore", ".devcontainer/Dockerfile",
    "assets/pretrained/model.safetensors", "assets/pretrained/config.json",
    "assets/pretrained/preprocessor_config.json", "assets/pretrained/source.json",
    "assets/pretrained/LICENSE.txt",
    *(f"notebooks/{name}" for name in NOTEBOOKS),
    *HF_FILES,
)
REQUIRED = {
    *FILES, ".devcontainer/devcontainer.json", ".vscode/settings.json",
    "requirements/codespaces.txt", "requirements/codespaces-jax.txt", "requirements/gpu.txt", "requirements/tpu.txt",
    "scripts/setup.sh", "scripts/doctor.py", "scripts/build_notebooks.py",
    "notebook_sources/__init__.py", "notebook_sources/common.py",
    "notebook_sources/data_and_inference.py", "notebook_sources/training.py",
    "notebook_sources/model_reference.py", ".agents/skills/vision-colab/SKILL.md",
}
EVIDENCE_FILES = {
    "student-review.json",
    *NOTEBOOKS, "notebook-checks.json", "report.json", "training.csv",
    "verification.json", "cleanup.json", "environment.json", "test-summary.json",
    "predictions.csv", "original_predictions.json", "confusion_head.csv",
    "confusion_finetune.csv", "learning_curves.png", "confusion_comparison.png",
}
HF_EVIDENCE_FILES = EVIDENCE_FILES | {
    *HF_NOTEBOOKS, *(name.replace(".ipynb", "_output.ipynb") for name in HF_NOTEBOOKS),
    "inference.json", "download_manifest.json", "preparation.json",
    "inference_report.json", "pretrained_top5.png", "environment.json", "test-summary.json",
}
EXCLUDED_PARTS = {
    "__pycache__", ".venv", ".cache", ".local-state", ".pytest_cache",
    ".ipynb_checkpoints", "runs", "results", "outputs", "custom_images", "presentation",
    "models", "finetuned_model",
}
PRIVATE_NAMES = {"token", "tokens", "credential", "credentials", "session", "sessions",
                 "auth", "oauth", "oauth2", "secret", "secrets"}


def eligible(path, root):
    """Never follow symlinks or accidentally include local state among source files."""
    relative = path.relative_to(root)
    if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name.startswith(".env") or path.stem.lower() in PRIVATE_NAMES:
        return False
    return not any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent)


def collect_files(root=ROOT):
    root = Path(root).resolve()
    files = {root / name for name in FILES if eligible(root / name, root)}
    for directory, suffixes in DIRECTORIES.items():
        for path in (root / directory).rglob("*"):
            if eligible(path, root) and path.suffix in suffixes:
                files.add(path)
    # The basic HF workshop keeps each public-library call inside its notebooks.
    # Only root-level guides, notebook builders and explicit dependency lists belong
    # to the student source bundle; downloaded model/data trees are never traversed.
    for pattern in ("*.md", "build*.py", "requirements*.txt"):
        for path in (root / "hf_colab_gpu").glob(pattern):
            if eligible(path, root):
                files.add(path)
    # Only this revision's compact validation evidence is distributed. Checkpoints,
    # terminal logs and old validation trees are intentionally outside this allowlist.
    for path in (root / "validation/direct").rglob("*"):
        if path.name in EVIDENCE_FILES and eligible(path, root):
            files.add(path)
    for path in (root / "hf_colab_gpu/validation").rglob("*"):
        if path.name in HF_EVIDENCE_FILES and eligible(path, root):
            files.add(path)
    missing = REQUIRED - {path.relative_to(root).as_posix() for path in files}
    if missing:
        raise FileNotFoundError("배포 필수 파일 누락: " + ", ".join(sorted(missing)))
    return sorted(files)


def validate_archive(path):
    with ZipFile(path) as archive:
        if archive.testzip():
            raise RuntimeError("배포 ZIP CRC 검증 실패")
        names = archive.namelist()
        relative = {name.removeprefix("vision_ai_notbook/") for name in names}
        if len(names) != len(set(names)) or REQUIRED - relative:
            raise RuntimeError("배포 ZIP 필수 파일 또는 중복 경로 검증 실패")
        for name in names:
            parts = Path(name).parts
            if not name.startswith("vision_ai_notbook/") or ".." in parts:
                raise RuntimeError(f"배포 ZIP 경로 오류: {name}")
            if any(part in EXCLUDED_PARTS for part in parts):
                raise RuntimeError(f"배포 제외 경로 발견: {name}")
            if name.startswith("vision_ai_notbook/validation/"):
                if not name.startswith("vision_ai_notbook/validation/direct/") or Path(name).name not in EVIDENCE_FILES:
                    raise RuntimeError(f"허용하지 않은 검증 파일: {name}")
            if name.startswith("vision_ai_notbook/hf_colab_gpu/validation/"):
                if Path(name).name not in HF_EVIDENCE_FILES:
                    raise RuntimeError(f"허용하지 않은 HF 검증 파일: {name}")
        if "pyproject.toml" in relative or any(name.startswith("vision_lab/") for name in relative):
            raise RuntimeError("이전 프로젝트 패키지가 ZIP에 포함됐습니다.")
    return len(names)


def build_distribution(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output)
    files = collect_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".tmp")
    try:
        with ZipFile(partial, "w", ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, Path("vision_ai_notbook") / path.relative_to(root))
        count = validate_archive(partial)
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    return {"file": str(output), "files": count, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "format": "HF CLI + Colab GPU basic workshop; JAX/TPU optional advanced notebooks; no project-package installation",
            "excludes": ["credentials", "private images", "data downloads", "run state",
                         "virtual environments", "raw logs", "validation checkpoints",
                         "legacy validation", "legacy Python package", "legacy slides"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "dist/vision_ai_notbook_hf_colab_gpu.zip")
    args = parser.parse_args()
    print(json.dumps(build_distribution(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
