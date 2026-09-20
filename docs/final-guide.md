# JAX·TPU 선택 심화 · 통합 강의 교안

기본 과정은 [HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)입니다. 이 문서는 JAX로 모델 연산을 살펴보고 GPU·TPU를 비교하는 선택 심화 과정의 교안과 기존 실측 기록입니다.

# JAX·TPU 선택 심화 · 학생 교안

이 문서는 JAX·TPU 선택 심화 과정입니다. [기본 HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)을 먼저 진행하세요. 심화 환경은 `bash scripts/setup.sh --with-jax`로 준비합니다.

이 실습에서는 모델을 불러와 이미지를 추론하고, 새로운 라벨 체계에 맞춰 학습한 뒤, 사전학습된 가중치 일부를 실제로 변경합니다. 최종 목표는 정확도 한 숫자를 높이는 데 그치지 않고 **어떤 데이터를 사용해 무엇을 학습했고, 어느 이미지에서 예측이 바뀌었는지 설명하는 것**입니다.

## 실습의 전체 흐름

```text
작업대 물체 이미지
    ↓ RGB 전처리
사전학습 DeiT의 원본 추론: ImageNet 1,000개 클래스
    ↓ 우리 과제에 맞는 5개 클래스와 데이터 분할
A. 사전학습 특징 고정 + 새 분류기 학습
    ↓ A의 학습 결과에서 시작
B. 마지막 Transformer 블록 + 최종 정규화 + 분류기 파인튜닝
    ↓ 동일한 테스트 이미지로 A와 B 비교
실패 사례 분석 → 직접 촬영한 사진으로 확장
```

Physical AI에서는 물체 종류가 뒤의 의사결정에 입력될 수 있습니다. 이 수업은 그 앞단의 **시각 인식 모델을 만들고 검증하는 과정**에 집중합니다. 이미지 전체에서 한 종류를 분류하므로 여러 물체의 위치 검출, 집기 좌표 추정, 로봇 동작은 수행하지 않습니다.

## 4시간 진행표

| 시간 | 활동 | 산출물 |
|---|---|---|
| 0:00–0:35 | Colab CLI, JAX, GPU·TPU 소개 | 실행 위치와 비용 구조 이해 |
| 0:35–0:55 | Codespaces 환경·데이터 점검 | 정상 환경 점검, 클래스 갤러리 |
| 0:55–1:20 | 사전학습 모델 추론 | 실제 이미지의 Top-5 예측 |
| 1:20–2:15 | GPU에서 분류기 학습과 파인튜닝 | GPU 보고서와 체크포인트 |
| 2:15–2:25 | 휴식 | 결과 회수·종료 상태 확인 |
| 2:25–3:00 | TPU에서 같은 실험 | GPU·TPU 비교 보고서 |
| 3:00–3:35 | 직접 촬영한 이미지 적용 | 데이터 폴더 또는 추가 실험 |
| 3:35–4:00 | 오분류 분석·발표·정리 | 결과 파일, 개선 제안 |

자원 할당과 설치 시간은 계정·네트워크에 따라 달라집니다. 기다리는 동안 학습 구조와 실패 사례 질문을 먼저 다룹니다.

## 1. 어디서 무엇이 실행되는가

Codespaces는 GitHub의 원격 개발 환경입니다. 브라우저에서 편집하더라도 파이썬은 Codespaces 안에서 실행됩니다. 00번·01번·04번은 Codespaces CPU에서 실행하고, 02번·03번은 코드를 읽고 수정한 뒤 공식 Colab CLI로 별도의 GPU·TPU 런타임에서 실행합니다. Codespaces CPU 커널에서 02번·03번을 실행한다고 원격 장치로 자동 연결되지는 않습니다.

가상환경은 라이브러리 버전을 분리하는 폴더입니다. 프로젝트를 설치 패키지로 묶지 않고 `numpy`, `PIL.Image`, `jax`, `optax` 등을 셀에서 직접 불러옵니다. 모델 계산과 학습 반복문도 셀에 정의되어 있습니다. [라이브러리 안내](libraries.md)에서 설치 이름과 import 이름을 확인하세요.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

이 출력에서 Python 실행 파일과 라이브러리 설치 경로를 확인합니다. 노트북 커널은 **Vision AI (Codespaces CPU)**를 선택하고 첫 셀의 `sys.executable`도 같은 가상환경을 가리키는지 봅니다.

JAX는 배열 연산, 자동 미분, 컴파일을 제공하는 파이썬 도구입니다. 이 프로젝트의 모델과 학습 코드는 JAX로 실행됩니다. `jax.jit`은 연산을 컴파일하고, `jax.value_and_grad`는 손실과 기울기를 계산합니다. GPU와 TPU는 이 연산을 실행할 수 있는 서로 다른 가속기입니다. 소스 코드를 공유하되 장치에 맞는 JAX 패키지를 설치합니다. [JAX 공식 문서](https://docs.jax.dev/en/latest/quickstart.html)

이 수업의 TPU 실행은 사용 가능한 장치 중 한 장치를 사용합니다. 다중 장치 분산 학습이나 GPU·TPU의 최대 처리량 경쟁을 목표로 하지 않습니다. 첫 컴파일 시간을 포함한 학습 시간만 보고 어느 장치가 항상 빠르다고 결론 내리지 마세요.

## 2. 데이터부터 확인하기

`00_setup_and_data.ipynb`를 실행합니다. 환경 설치는 라이브러리만 준비하며, 데이터 다운로드·전처리·분할 저장은 이 노트북의 셀에서 직접 진행합니다. 기본 클래스는 `bottle`, `bowl`, `can`, `cup`, `plate`입니다. 클래스별 학습 100장·검증 20장·테스트 40장이므로 총 500/100/200장입니다. 준비한 뒤 `python scripts/doctor.py --data`로 파일의 체크섬을 검사합니다.

학습 데이터는 가중치 업데이트에, 검증 데이터는 epoch 선택에, 테스트 데이터는 선택이 끝난 모델의 평가에 사용합니다. 분할 파일과 식별자의 중복을 검사하고 체크섬을 기록합니다. CIFAR-100의 작은 공개 이미지는 실습용 출발점이며, 여러분의 카메라·배경·조명에서의 성능을 대신하지 않습니다. [CIFAR 원문](https://www.cs.toronto.edu/~kriz/cifar.html), [데이터 배포 저장소](https://huggingface.co/datasets/uoft-cs/cifar100)

관찰할 내용:

- 컵과 그릇을 구분할 때 손잡이·높이·테두리가 충분히 보이는가?
- 배경만으로 정답을 짐작할 수 있는 이미지가 있는가?
- 32×32 이미지를 확대하면 원래 없던 세부 정보까지 생기는가?

## 3. 원본 모델로 먼저 추론하기

`01_pretrained_inference.ipynb`에서 실제 `facebook/deit-tiny-patch16-224` 체크포인트를 읽습니다. 모델은 ImageNet의 1,000개 클래스에 대해 학습된 DeiT입니다. 입력은 224×224이며 16×16 패치로 나누어 처리합니다. 이 프로젝트는 원본 파라미터를 JAX 연산에 그대로 대응시키고, 해당 고정 버전의 전처리 설정을 따릅니다. [모델 자료](https://huggingface.co/facebook/deit-tiny-patch16-224)

Top-5 라벨과 점수를 보고 이미지와 맞는지 설명합니다. 원본 모델의 라벨에는 우리가 만든 5개 클래스와 이름·범위가 다른 항목이 섞여 있습니다. 따라서 원본 1,000개 클래스의 결과와 새 5개 클래스의 정확도를 같은 기준처럼 직접 비교하지 않습니다.

이 단계는 **Codespaces CPU의 실제 추론**입니다. GPU·TPU 학습을 수행했다는 뜻은 아닙니다.

## 4. 새 분류기를 학습하기: 비교 기준 만들기

기존 1,000개 클래스 분류기를 5개 클래스 분류기로 교체합니다. 새 분류기는 처음에는 무작위 가중치이므로 바로 출력한 정확도를 ‘사전학습 모델의 정확도’라고 부르지 않습니다.

먼저 사전학습 백본을 고정하고 새 분류기만 학습합니다. 이를 이 수업의 **head 기준 모델**이라고 합니다. 사전학습 특징이 이미 우리 물체를 얼마나 잘 구분하는지 확인하는 단계입니다. 학습률은 0.001, 기본 학습량은 3 epoch입니다. 매 epoch 검증 손실을 계산하고 가장 낮은 epoch를 저장합니다.

학습은 클래스 정답과 모델 출력의 차이를 줄이도록 가중치를 업데이트하는 과정입니다. 손실이 감소해도 검증 정확도가 같은 폭으로 오르지 않을 수 있습니다. 손실은 정답에 준 확률까지 반영하고, 정확도는 가장 높은 점수의 라벨이 맞았는지만 셉니다.

## 5. 마지막 블록을 실제로 파인튜닝하기

다음은 방금 학습한 분류기에서 시작하여 **마지막 Transformer 블록·최종 LayerNorm·분류기**를 함께 업데이트합니다. 앞의 11개 Transformer 블록은 고정하며, 출력 토큰을 캐시해 반복 연산을 줄입니다. 기본 파인튜닝 학습률은 0.0001, 학습량은 3 epoch입니다.

이 과정은 분류기만 바꾸는 단계와 다릅니다. `report.json`의 `last_block_changed`와 `last_block_delta_l2`가 사전학습 블록의 실제 변경을 보여줍니다. `frozen_prefix_unchanged`는 고정하기로 한 앞부분이 바뀌지 않았는지 확인합니다. 전체 모델을 처음부터 학습하거나 모든 블록을 업데이트하는 실습은 아닙니다.

캐시를 재사용하기 위해 이번 기본 실습에는 무작위 이미지 증강을 넣지 않았습니다. 후속 실험에서 증강을 추가하려면 증강 후 이미지에 맞춰 특징을 다시 계산하는 방식도 함께 설계해야 합니다.

## 6. Colab GPU 실행

최초 인증은 터미널에서 진행합니다. 공식 CLI 0.6.0의 `sessions` 명령이 필요한 경우 Google 로그인을 안내합니다.

```bash
source .venv/bin/activate
colab --auth oauth2 sessions
```

[Colab CLI 직접 실행 안내](direct-colab-cli.md)의 GPU 준비 단계를 따라 새 세션 생성, `requirements/gpu.txt` 설치, 커널 재시작, 입력 파일 전송을 각각 진행합니다. 모델·데이터 파일은 자동으로 동기화되지 않으므로 명시적으로 올립니다. 준비가 끝난 뒤 다음 명령을 실행합니다.

```bash
colab exec -s vision-gpu -f notebooks/02_gpu_finetuning.ipynb --timeout 1800
```

02번에서 읽은 코드 셀이 그대로 Colab에서 실행됩니다. `jax.value_and_grad`가 미분하고 `optax.adam`과 `optax.apply_updates`가 가중치를 갱신하는 부분을 따라가 보세요. 실행 기록은 Codespaces의 `notebooks/02_gpu_finetuning_output.ipynb`에 저장됩니다.

직접 실행 안내의 `download` 명령으로 결과를 회수합니다. `results/gpu/report.json`의 `status`가 `completed`이고 실제 장치가 `gpu`인지 확인합니다. 출력 노트북에도 오류가 없어야 합니다. 결과 파일만 있거나 CLI 명령이 끝났다는 이유만으로 성공이라고 판단하지 않습니다. 회수 후 `colab stop -s vision-gpu`, `colab sessions`로 종료를 확인합니다.

## 7. 무엇이 달라졌는지 평가하기

같은 테스트 200장에서 head 기준 모델과 파인튜닝 모델을 비교합니다.

| 지표·파일 | 읽는 방법 |
|---|---|
| Accuracy | 전체 이미지 중 맞힌 비율 |
| Macro F1 | 클래스별 F1의 평균. 어느 한 클래스의 실패도 확인 |
| `training.csv` | 단계별 학습·검증 손실과 검증 정확도 |
| `predictions.csv` | 같은 이미지에서 예측이 어떻게 바뀌었는지 |
| 혼동행렬 | 행은 실제 클래스, 열은 예측 클래스 |
| 체크포인트 | 검증 손실로 선택한 모델 가중치 |

노트북에서 손실 곡선과 혼동행렬을 보고 세 가지 사례를 찾습니다. ‘둘 다 정답’, ‘파인튜닝 후 정답’, ‘파인튜닝 후 오답’입니다. 마지막 경우도 중요한 결과입니다. 표본이 작거나 학습률이 맞지 않으면 파인튜닝이 더 나쁠 수 있습니다.

검증 결과로 개선 방향을 정한 뒤 이전 결과를 별도 폴더에 보관하고 새 세션에서 실험하세요. 노트북의 `OUTPUT_DIR`을 바꿨다면 내려받을 원격 경로도 함께 바꿉니다. 테스트 결과를 계속 보며 설정을 고르면 테스트 세트도 사실상 튜닝 데이터가 됩니다.

## 8. TPU에서 재현하기

`03_tpu_and_compare.ipynb`에서 GPU와 같은 데이터와 학습 설정을 사용합니다. 직접 실행 안내의 TPU 단계대로 새 세션을 만들고 `requirements/tpu.txt`를 설치한 뒤 커널을 다시 시작합니다. 입력 파일과 회수한 GPU의 `report.json`도 TPU에 전송합니다. 준비 후 실행할 명령은 다음과 같습니다.

```bash
colab exec -s vision-tpu -f notebooks/03_tpu_and_compare.ipynb --timeout 1800
```

GPU와 TPU는 같은 초기 모델과 데이터에서 각각 독립적으로 학습합니다. 03번 마지막 비교 셀은 모델·데이터·클래스 순서·분할 ID·학습 설정을 검사하고 조건이 같을 때 결과를 비교합니다. 부동소수점 계산 차이로 결과가 완전히 같을 필요는 없습니다. 실제 장치 기록, 가중치 변경, 지표를 함께 확인하세요.

TPU 보고서를 `results/tpu/report.json`으로 회수하고 `status=completed`, `device.platform=tpu`인지 확인합니다. 출력 노트북의 오류 여부와 세션 종료도 별도로 확인합니다. CPU 검증의 `cpu_validation`, `cpu_smoke_only`는 GPU·TPU 실습 완료가 아닙니다.

## 9. 직접 촬영한 작업대 사진으로 확장하기

`04_custom_images.ipynb`를 사용합니다. 하나의 이미지에는 분류하려는 주된 물체가 명확하게 보여야 합니다. 클래스 이름은 자유롭게 정할 수 있으며 각 분할에서 같은 이름을 사용합니다.

```text
custom_images/
  train/bottle/*.jpg
  train/cup/*.jpg
  validation/bottle/*.jpg
  validation/cup/*.jpg
  test/bottle/*.jpg
  test/cup/*.jpg
```

같은 동영상의 연속 프레임을 무작위로 나누면 거의 같은 장면이 학습과 테스트에 함께 들어갈 수 있습니다. 촬영 세션·날짜·배경을 기준으로 분리하고 물체별 조명·방향·거리도 바꿔 보세요.

04번에서 `CUSTOM_INPUT`을 사진 폴더로 지정하고 `PREPARE_CUSTOM=True`로 바꿉니다. Pillow로 사진을 읽고 NumPy 배열로 저장하는 셀을 실행하면 `CUSTOM_DATA`에 네 파일이 생깁니다. 새 데이터로 학습하려면 직접 실행 안내의 업로드 명령에서 **로컬 입력 경로만** `data/my-workbench/manifest.json`, `train.npz`, `validation.npz`, `test.npz`로 바꿉니다. 원격 경로는 `content/vision-ai/data/prepared/...`로 유지하면 02번·03번의 `DATA_PATH`를 바꾸지 않아도 됩니다. 사진은 이 전송 단계를 실행할 때 본인의 Colab 런타임으로 올라갑니다.

회수한 새 체크포인트로 한 장을 추론하려면 04번의 `CHECKPOINT`와 `IMAGE_PATH`를 지정한 뒤 아래 추론 셀을 실행합니다. `safetensors`로 원본 가중치를 읽고 학습한 부분을 복원한 다음, Pillow 전처리·JAX 추론·Matplotlib 표시를 차례대로 확인합니다.

실물 연결을 고민할 때에는 처음 보는 물체와 불확실한 예측을 별도로 검증해야 합니다. 이 분류기의 softmax 최고 점수를 곧바로 실제 동작의 안전성으로 해석하지 않습니다.

## 10. 결과 보관과 종료

`results/gpu`, `results/tpu`의 보고서·체크포인트·CSV와 `notebooks/*_output.ipynb`를 보관합니다. 결과 폴더와 출력 노트북을 Codespaces 탐색기에서 내 컴퓨터로 내려받으세요. 이번 실습에서 만든 Colab 세션은 `colab stop -s <세션 이름>`으로 종료하고 `colab sessions`로 확인합니다. 학습이나 다운로드가 실패했어도 종료 확인은 생략하지 않습니다.

결과를 내려받은 뒤 Codespaces도 중지합니다. Codespaces 사용료와 Colab CU는 별개이며, CU는 원하는 GPU의 무제한 사용을 보장하지 않습니다. [Codespaces 비용](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces), [Colab 사용 제한](https://research.google.com/colaboratory/faq.html)

제출물에는 두 장치의 실제 보고서, 학습 전후 오분류 사례 3개, 직접 촬영한 데이터의 분할 기준, 다음 실험에서 바꿀 조건 한 가지를 포함합니다. 이번 직접 라이브러리 버전의 실행 여부와 측정값은 [현재 검증 기록](direct-test-results.md)에서 확인합니다.


---

# JAX·TPU 심화 · 직접 호출 버전의 실제 검증 기록

이 문서는 JAX·TPU 선택 심화 과정입니다. [기본 HF·PyTorch GPU 실습](../hf_colab_gpu/README.md)을 먼저 진행하세요. 심화 환경은 `bash scripts/setup.sh --with-jax`로 준비합니다.

검증일: 2026-09-19. 자체 Python 패키지를 설치하지 않고, 학생 노트북에 보이는 코드 셀을 실행한 결과입니다. 이전 버전의 GPU·TPU 실행 결과와 구분합니다.

## 무엇을 실행했나요?

NumPy, Pillow, Safetensors, JAX, Optax, Matplotlib, PyArrow를 직접 불러옵니다. 데이터 준비, 모델 연산, 손실 함수, 두 학습 반복문, 평가와 저장 코드가 노트북 안에 있습니다.

프로젝트 패키지가 설치되지 않았고 `vision_lab`를 import할 수 없는 상태임을 확인했습니다. 별도의 새 Python 3.12 가상환경에서도 requirements의 공개 라이브러리만 설치해 import와 JAX CPU 검사를 통과했습니다. [환경 근거](../validation/direct/environment.json)

## 노트북 실행

| 노트북 | 수행한 검사 | 결과 |
|---|---|---|
| 00 · 환경과 데이터 | 실제 데이터 읽기·분할·저장, 샘플 그림 | 통과 |
| 01 · 사전학습 추론 | 실제 가중치, 이미지 전처리, CPU 추론, Top-5 그림 | 통과 |
| 02 · GPU 학습 코드 | CPU를 명시해 전체 500/100/200장, 각 3 epoch 실행 | 통과; GPU 검증과 별도 |
| 03 · TPU 학습 코드 | CPU 축소 모드를 명시해 셀 실행 흐름 검사 | 통과; TPU 검증과 별도 |
| 04 · 직접 촬영한 사진 | 기본 안내와 입력 유무 확인 분기 | 통과; 사용자 사진은 제공되지 않음 |

실행한 노트북은 [validation/direct](../validation/direct)에 출력과 함께 보관했습니다. [노트북별 검사 기록](../validation/direct/notebook-checks.json)

## 같은 조건의 전체 학습 결과

- 모델: `facebook/deit-tiny-patch16-224`, revision `b3428f18dcc7b543470d07f14b4a4157815d1880`
- 물체: 병·그릇·캔·컵·접시. CIFAR-100에서 학습 500장, 검증 100장, 테스트 200장
- 분류기 학습 3 epoch, 학습률 0.001 → 마지막 블록 파인튜닝 3 epoch, 학습률 0.0001
- JAX 0.11.1, FP32, Adam, batch size 16, seed 42
- 검증 손실로 각 단계의 체크포인트를 선택하고 테스트 정답은 최종 평가에만 사용

| 실행 장치 | 분류기만 학습한 정확도 | 부분 파인튜닝 후 정확도 | 파인튜닝 후 macro-F1 |
|---|---:|---:|---:|
| 로컬 CPU · 전체 데이터 | 79.0% | 82.0% | 0.8204 |
| Colab Tesla T4 GPU | 79.0% | 82.0% | 0.8204 |
| Colab TPU v5 lite · 요청 기종 v5e1 | 79.0% | 82.0% | 0.8204 |

GPU와 TPU는 새 노트북을 공식 Colab CLI로 각각 실행했습니다. 두 실행 모두 셀 오류가 없고 결과 파일의 SHA-256이 보고서와 일치하며, 앞부분 가중치 불변과 마지막 블록 변경을 확인했습니다. TPU 노트북에서는 업로드한 GPU 보고서와 모델·데이터·설정이 같은지 검사한 뒤 비교 그림을 출력했습니다. 결과 회수 후 각 세션 종료와 서버의 활성 세션 없음까지 확인했습니다.

- [이번 GPU 전체 보고서](../validation/direct/gpu/report.json)
- [GPU 코드 실행·파일 검증](../validation/direct/gpu/verification.json)
- [GPU 세션 종료 기록](../validation/direct/gpu/cleanup.json)
- [이번 TPU 전체 보고서](../validation/direct/tpu/report.json)
- [TPU 코드 실행·파일 검증](../validation/direct/tpu/verification.json)
- [TPU 세션 종료 기록](../validation/direct/tpu/cleanup.json)

원본 1,000개 ImageNet 라벨 추론은 관찰용입니다. 표의 비교 기준은 새 5개 클래스 분류기를 학습한 모델입니다. 200장 중 정답이 158장에서 164장으로 늘었으며 개선 폭은 3%p입니다. 이번 작은 공개 데이터 분할에서 얻은 값으로, 실제 작업대 카메라의 성능을 뜻하지 않습니다.

CPU·GPU·TPU 전체 실행에서 앞의 11개 블록 등 동결 부분은 그대로이고 마지막 사전학습 블록은 변경됐습니다. 학습 대상은 마지막 블록·최종 정규화·분류기의 총 446,213개 파라미터입니다. [CPU 전체 보고서](../validation/direct/cpu-full/report.json)

![실제 GPU의 학습 단계별 테스트 혼동행렬](../validation/direct/gpu/confusion_comparison.png)

## 검증 범위와 재실행

환경·패딩 배치의 손실/미분·배포 파일 검사를 포함한 자동 테스트 15개를 통과했습니다. 검증용 코드도 별도 프로젝트 패키지 설치 없이 실행하며, `python -m pytest -q` 또는 `pytest -q`를 사용할 수 있습니다.

배포 ZIP을 다른 임시 폴더에 풀고, 별도로 만든 공개 라이브러리 전용 가상환경에서 01번 추론 셀을 처음부터 끝까지 실행했습니다. 기존에 검증한 준비 데이터를 입력으로 복사했으며 프로젝트 경로와 자체 패키지에 의존하지 않는 것을 확인했습니다. 이 검사는 네트워크 다운로드나 Codespaces 컨테이너 빌드 검사를 대신하지 않습니다. [검사 요약](../validation/direct/test-summary.json)

04번은 실제 사용자 사진 대신 공개 이미지에서 별도로 고른 두 클래스의 소규모 입력으로 사진 읽기·분할 저장·중복 검사·덮어쓰기 거부를 확인했습니다. 이 두 클래스 데이터로 02번의 학습 반복문을 실행하고, 04번에서 체크포인트를 복원해 추론하는 과정도 통과했습니다. 기본 5개 클래스 체크포인트의 복원 예측은 저장된 예측 CSV와 일치했습니다. 갤러리·Top-5·혼동행렬·학습 곡선·복원 추론 그림도 직접 확인했습니다. [추가 검사 기록](../validation/direct/student-review.json)

실제 GitHub Codespace를 새로 생성해 최초 컨테이너 빌드부터 확인한 것은 아닙니다. 이 호스트에서는 Docker를 실행할 수 없어 새 가상환경 설치와 노트북 실행을 확인했습니다. 학생 계정의 첫 OAuth2 로그인과 GPU·TPU 가용량도 별도로 확인해야 합니다.

노트북 00·01·04는 Codespaces CPU에서 실행하고, 02·03은 [공식 Colab CLI 단계별 안내](direct-colab-cli.md)에 따라 각각 GPU·TPU에서 실행합니다. 출력 노트북의 오류 유무, 실제 장치, 보고서의 완료 상태, 가중치 변경, 결과 회수, 세션 종료까지 확인합니다.

GPU·TPU 노트북의 기본값은 해당 가속기입니다. 가속기를 찾지 못하면 중단하며 CPU 결과로 완료를 대신하지 않습니다. 강사용 CPU 검증은 `VISION_DEVICE=cpu`를 명시했을 때만 진행됩니다.
