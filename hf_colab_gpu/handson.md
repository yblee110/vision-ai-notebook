# 사전학습 모델을 받아 GPU에서 추론하고 파인튜닝하기

오늘 만들 것은 병·그릇·캔·컵·접시를 구분하는 이미지 분류기입니다. Physical AI의 앞단에서 물체 종류를 알려 주는 데 쓸 수 있지만, 이번 실습에서는 로봇 제어와 집기 좌표 계산까지 다루지 않습니다.

모델 구조는 Transformers가 불러옵니다. 학생이 집중할 부분은 **어떤 모델과 데이터를 받았는지, GPU에서 어떻게 실행하는지, 추가 학습으로 무엇이 달라지는지**입니다.

## 진행 순서

| 시간 | 활동 | 확인할 내용 |
|---|---|---|
| 0–35분 | Colab CLI·HF CLI·모델·가속기 소개 | 어디서 파일을 받고 어디서 연산하는가 |
| 35–55분 | Codespaces 준비와 00번 노트북 | 모델 3개 파일, 데이터 500/100/200장 |
| 55–85분 | Colab GPU 연결·전송·01번 추론 | 실제 GPU 이름과 원본 모델 Top-5 |
| 85–135분 | 02번 분류기 학습·파인튜닝 | 같은 테스트 이미지의 예측 변화 |
| 135–155분 | 저장·복원·결과 회수·종료 | 모델 재로딩과 GPU 세션 종료 |
| 선택 30–45분 | JAX·TPU 심화 | 별도 JAX 노트북에서 같은 과제 살펴보기 |

처음 설치하거나 자원을 기다리는 시간은 계정·네트워크에 따라 달라집니다. 기본 설명 35분 중 JAX·TPU 소개는 합쳐 5분 안팎으로 두고 나머지는 CLI와 이번 실습 흐름에 씁니다.

## 두 CLI와 두 실행 환경 구분하기

`hf download`는 Hugging Face Hub에서 모델 파일을 내려받습니다. `colab`은 Colab 런타임을 만들고 그 안에서 Python 코드를 실행합니다. 하나는 파일을 받는 도구이고 다른 하나는 원격 컴퓨터를 다루는 도구입니다.

Codespaces에서는 노트북 편집·모델 다운로드·데이터 준비를 합니다. GPU 추론과 학습은 Colab에서 수행합니다. Colab GPU를 만들었다고 Codespaces의 Python 커널이 GPU 커널로 바뀌지는 않습니다. `colab exec -f`로 노트북의 코드 셀을 Colab에 보내야 합니다.

Colab CU와 Codespaces 사용료는 별개입니다. Colab CU 잔액·할당 가능 여부를 확인하고, 실습 후 직접 만든 세션을 종료합니다. Runpod이나 GCP와 비용 구조를 비교할 때에도 ‘과금이 절대 발생하지 않는다’고 설명하지 않습니다. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## 00 · 모델 다운로드와 데이터 준비

`hf download`의 네 요소를 확인합니다. 모델 ID, 받을 파일 이름, 고정 revision, 저장할 폴더입니다. revision을 고정하면 같은 모델 버전을 다시 받을 수 있습니다. 파일 SHA-256 검사까지 마친 뒤 다음 단계로 갑니다.

공개 모델 `facebook/deit-tiny-patch16-224`는 ImageNet 1,000개 클래스로 사전학습됐습니다. 이 모델은 Transformers의 ViT 분류 모델로 불러옵니다. 이번 고정 revision의 원본 가중치는 `pytorch_model.bin`이며, 학습 후 저장하는 파일은 Safetensors입니다. 형식이 다른 것은 다운로드 실패가 아닙니다.

데이터는 학습·검증·테스트로 나눕니다. 학습은 가중치 업데이트, 검증은 저장할 모델 선택, 테스트는 최종 성능 확인에 사용합니다. CIFAR-100의 원본 이미지는 32×32입니다. 모델 입력인 224×224로 늘려도 새로운 세부 정보가 생기지는 않습니다.

실행할 파일: [00_hf_download_and_data.ipynb](notebooks/00_hf_download_and_data.ipynb)

## 01 · GPU 연결과 원본 추론

[단계별 실행 안내](README.md)에 따라 Google 로그인 → T4 생성 → 라이브러리 설치 → 커널 재시작 → 파일 전송을 진행합니다. Colab 커널에서 `torch.cuda.is_available()`과 장치 이름을 확인합니다.

모델을 불러오는 핵심 코드는 다음 두 호출입니다.

```python
processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
model = AutoModelForImageClassification.from_pretrained(model_dir, local_files_only=True)
```

실제 노트북에는 원본 가중치 형식과 GPU 지정도 명시돼 있습니다. `processor`는 이미지 크기와 정규화를 처리하고, `model`은 클래스 점수를 계산합니다. `torch.inference_mode()`는 추론에 필요 없는 기울기 계산을 끕니다.

Top-5를 보고 이미지와 맞는지 설명합니다. 원본 모델의 라벨 이름이 우리 다섯 클래스와 같지 않을 수 있습니다. 원본 1,000개 라벨의 출력과 새 5개 클래스의 정확도는 같은 기준이 아닙니다.

실행할 파일: [01_gpu_inference.ipynb](notebooks/01_gpu_inference.ipynb)

## 02 · 비교 기준을 학습하고 일부 가중치를 수정하기

먼저 기존 모델의 마지막 분류기를 5개 클래스 분류기로 바꿉니다. 새 분류기의 가중치는 무작위 상태이므로 그것을 곧바로 사전학습 성능이라고 부르지 않습니다. 백본을 고정하고 새 분류기만 3 epoch 학습한 결과를 비교 기준으로 사용합니다.

그다음 학습한 분류기에서 이어서 마지막 Transformer 블록·최종 LayerNorm·분류기를 3 epoch 더 학습합니다. 앞의 11개 블록은 고정합니다. 노트북에서 학습 대상으로 정한 파라미터를 출력하고, 학습 전후 파일 지문으로 실제 변경 범위를 검사합니다.

```python
optimizer.zero_grad(set_to_none=True)
logits = model(pixel_values=pixels).logits
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

이 다섯 줄을 중심으로 설명합니다. 예측과 정답의 차이를 손실로 만들고, 기울기를 구해 가중치를 갱신합니다. 모델 내부 Attention을 직접 구현하지 않아도 데이터·학습률·학습 범위를 바꾸는 실험을 할 수 있습니다.

저장할 epoch는 **검증 정확도가 높은 순서, 동률이면 검증 손실이 낮은 순서**로 고릅니다. 테스트 데이터는 이 선택에 사용하지 않습니다. 각 단계에서 선택을 마친 모델을 같은 테스트 200장으로 평가합니다.

실행할 파일: [02_gpu_finetuning.ipynb](notebooks/02_gpu_finetuning.ipynb)

## 결과를 해석하고 다시 사용하기

학습 곡선에서는 검증 성능이 어떻게 움직였는지 봅니다. 혼동행렬에서는 어떤 종류를 서로 헷갈렸는지 확인합니다. `predictions.csv`에서 ‘둘 다 정답’, ‘파인튜닝 후 정답’, ‘파인튜닝 후 오답’ 사례를 하나씩 고릅니다. 파인튜닝이 모든 클래스에서 개선되리라고 가정하지 않습니다.

`save_pretrained()`로 모델 설정·가중치·전처리 설정을 저장한 뒤 같은 폴더를 다시 불러옵니다. 노트북은 저장 전후 같은 이미지의 출력이 일치하는지 확인합니다. 실제 작업대 사진에 적용하려면 촬영 환경에 맞는 데이터를 따로 수집하고 평가해야 합니다.

학습 결과 파일과 출력 노트북을 회수한 뒤 Colab 세션을 종료합니다. 브라우저 창이나 Codespaces를 닫는 것만으로 Colab 세션 종료를 대신하지 않습니다.

## 제출물과 선택 심화

제출할 자료는 실제 GPU 이름, 원본 추론 결과, 파인튜닝 전후 지표·오분류 사례, 저장 모델 재로딩 결과, 결과 회수와 세션 종료 기록입니다. 수업 예시의 수치를 본인의 실행 결과로 복사하지 않습니다.

JAX·TPU는 [별도 심화 안내](../docs/jax-advanced.md)에서 이어갑니다. 기본 과정의 PyTorch 체크포인트와 심화의 JAX 체크포인트는 형식이 다르며, 두 구현의 수치를 단순한 프레임워크 성능 비교로 사용하지 않습니다.
