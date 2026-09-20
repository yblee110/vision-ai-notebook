"""Combine the basic course handout with its own measured verification record."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
intro = "# HF CLI와 Colab GPU · 최종 강의 교안\n\nTransformers·PyTorch로 추론·파인튜닝하는 기본 과정입니다. JAX·TPU는 선택 심화로 다룹니다.\n\n"
parts = [(HERE / name).read_text(encoding="utf-8") for name in ("handson.md", "test-results.md")]
(HERE / "final-guide.md").write_text(intro + "\n\n---\n\n".join(parts), encoding="utf-8")
