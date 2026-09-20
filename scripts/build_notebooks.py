"""Maintainer tool: embed every lesson operation as visible notebook cells."""
from pathlib import Path
import ast
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from notebook_sources.data_and_inference import make_notebooks as data_notebooks
from notebook_sources.training import make_notebooks as training_notebooks

books = {**data_notebooks(), **training_notebooks()}
assert len(books) == 5
for name, book in sorted(books.items()):
    book.cells[0].source = book.cells[0].source.replace('# ', '# [JAX·TPU 선택 심화] ', 1)
    book.cells[0].source += '\n\n기본 HF·PyTorch 과정은 `hf_colab_gpu/notebooks`에 있습니다. 이 심화 과정은 프로젝트 최상위에서 `bash scripts/setup.sh --with-jax`로 준비합니다.'
    for cell in book.cells:
        if cell.cell_type != 'code':
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(('vision_lab', 'notebook_sources')) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or '').startswith(('vision_lab', 'notebook_sources'))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {'exec', 'eval', 'run_training', 'fit_stage'}
        assert 'scripts/colab_lab.py' not in cell.source
    nbformat.validate(book)
    path = ROOT / 'notebooks' / name
    nbformat.write(book, path)
    print(f'{name}: {len(book.cells)} cells, all lesson code embedded')
