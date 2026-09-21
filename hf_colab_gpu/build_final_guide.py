"""Combine the basic course handout with its own measured verification record."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
intro = "# HF CLI와 Colab GPU · 최종 강의 교안\n\n`notebooks`의 00·01·02번으로 진행하는 Transformers·PyTorch 실습입니다. 01·02번에는 각각 두 개의 AI 코딩 빈칸이 있습니다. 뒤의 실측 결과는 2026-09-19 완성된 코드의 기록이며 학생이 작성한 코드의 실행 결과와 구분합니다.\n\n"
parts = [(HERE / name).read_text(encoding="utf-8") for name in ("handson.md", "test-results.md")]
(HERE / "final-guide.md").write_text(intro + "\n\n---\n\n".join(parts), encoding="utf-8")
