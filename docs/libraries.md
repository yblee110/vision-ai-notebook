# JAX·TPU 선택 심화 · 라이브러리 안내

이 문서는 JAX·TPU 선택 심화 과정입니다. [기본 HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)을 먼저 진행하세요. 심화 환경은 `bash scripts/setup.sh --with-jax`로 준비합니다.

이 수업에서는 공개 라이브러리를 직접 import합니다. 모델 구조, 추론, 손실 계산, 미분, 가중치 갱신 코드는 노트북 셀에서 읽을 수 있습니다. `.venv`는 라이브러리 버전을 다른 프로젝트와 분리해 두는 폴더입니다. 수업 코드를 하나의 설치 패키지로 감추는 기능은 없습니다.

## 설치 이름과 import 이름

| 설치 이름 | 코드에서 부르는 이름 | 수업에서 하는 일 |
|---|---|---|
| `numpy==2.5.2` | `import numpy as np` | 이미지 배열, 데이터 분할, `.npz` 파일 읽기·쓰기 |
| `Pillow==12.3.0` | `from PIL import Image` | 이미지 파일 열기, RGB 변환, 크기 조절 |
| `safetensors==0.8.0` | `from safetensors.numpy import load_file` | 배포된 사전학습 가중치를 NumPy 배열로 읽기 |
| `jax==0.11.1` | `import jax`, `import jax.numpy as jnp` | 모델 연산, `jax.jit` 컴파일, `jax.value_and_grad` 자동 미분 |
| `optax==0.2.8` | `import optax` | 손실 함수와 `optax.adam`, `optax.apply_updates`로 가중치 갱신 |
| `matplotlib==3.11.1` | `import matplotlib.pyplot as plt` | 이미지, 학습 곡선, 혼동행렬 그리기 |
| `pyarrow==20.0.0` | `import pyarrow.parquet as pq` | 원본 이미지가 담긴 Parquet 파일 읽기 |

GPU에서는 `jax[cuda12]==0.11.1`, TPU에서는 `jax[tpu]==0.11.1`을 설치합니다. 대괄호는 해당 장치에 필요한 추가 의존성을 뜻하며, 코드는 두 경우 모두 `import jax`로 시작합니다. GPU와 TPU용 목록은 각각 별도 Colab 런타임에 설치합니다.

```python
import numpy as np
from PIL import Image
from safetensors.numpy import load_file
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

print(jax.devices())  # 이 셀을 실행한 컴퓨터의 장치를 확인합니다.
```

각 노트북은 필요한 라이브러리만 import합니다. 예를 들어 PyArrow는 Codespaces의 데이터 준비에 쓰이며 GPU·TPU 학습에는 필요하지 않습니다.

## 직접 읽을 수 있는 연산

```python
# NumPy 배열을 JAX 배열로 바꿉니다.
images = np.zeros((2, 224, 224, 3), dtype=np.float32)
images_on_device = jnp.asarray(images)

# Adam 옵티마이저를 만듭니다.
optimizer = optax.adam(learning_rate=1e-4)
```

위 코드는 라이브러리의 역할을 확인하는 짧은 예시입니다. 실제 모델의 파라미터, 손실 함수와 학습 반복문은 02번·03번 노트북에 정의되어 있습니다. `from vision_lab ...`, `python -m vision_lab ...` 같은 프로젝트 전용 호출 없이 셀을 위에서부터 읽고 실행합니다.

## 노트북·CLI를 위한 도구

| 설치 이름 | 역할 |
|---|---|
| `ipykernel==7.3.0` | VS Code가 선택한 Python으로 셀을 실행하도록 연결 |
| `nbformat==5.10.4` | `.ipynb` 형식을 읽고 검사 |
| `nbclient==0.10.4` | 노트북 자동 실행 검증 |
| `pytest==9.1.1` | 수업 코드와 환경의 검증 |
| `google-colab-cli==0.6.0` | 터미널의 `colab` 명령 제공 |
| `jupyter-kernel-client==0.9.0` | CLI 0.6.0이 원격 Python 커널과 통신할 때 사용하는 호환 버전 |

이 도구들은 이미지 분류 모델의 구성요소가 아닙니다. 학생이 모델을 읽는 데 필요한 import와 환경을 운영하는 도구를 나누어 봅니다.

## 설치하지 않아도 되는 Python 표준 라이브러리

`pathlib`, `json`, `io`, `csv`, `hashlib`, `time`, `urllib.request`, `sys`, `os`는 Python에 들어 있습니다. 파일 경로, 설정, 다운로드, 체크섬, 시간 측정 등에 사용하며 따로 `pip install`하지 않습니다.

## 지금 어느 Python과 라이브러리를 쓰는지 확인하기

```bash
source .venv/bin/activate
python scripts/doctor.py
```

출력에는 Python 실행 파일 경로, 각 라이브러리의 버전, import 이름과 실제 설치 파일 경로가 나옵니다. 설치 목록은 다음 세 파일에 한 줄씩 적혀 있습니다.

- `requirements/codespaces.txt`: Codespaces CPU와 수업 도구
- `requirements/gpu.txt`: Colab GPU 학습
- `requirements/tpu.txt`: Colab TPU 학습

프로젝트를 설치하는 `pip install -e .`는 사용하지 않습니다. 00번 노트북에서 데이터를 준비한 뒤에는 `python scripts/doctor.py --data`로 데이터 파일의 체크섬도 확인할 수 있습니다.
