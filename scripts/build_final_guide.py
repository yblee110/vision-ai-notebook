"""Combine the student handout and current measured results for the instructor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
intro = "# JAX·TPU 선택 심화 · 통합 강의 교안\n\n기본 과정은 [HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)입니다. 이 문서는 JAX로 모델 연산을 살펴보고 GPU·TPU를 비교하는 선택 심화 과정의 교안과 기존 실측 기록입니다.\n\n"
parts = [(ROOT / "docs" / name).read_text(encoding="utf-8") for name in ("handson.md", "direct-test-results.md")]
(ROOT / "docs/final-guide.md").write_text(intro + "\n\n---\n\n".join(parts), encoding="utf-8")
