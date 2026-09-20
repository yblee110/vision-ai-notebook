# 실습 중 막혔을 때

## 어느 Python과 라이브러리를 쓰는지부터 확인하기

`.vision-lab-root`가 있는 프로젝트 폴더에서 다음 명령을 실행합니다.

```bash
source .venv/bin/activate
python scripts/doctor.py
```

출력에는 Python 실행 파일과 각 라이브러리의 버전·import 이름·설치 경로가 나옵니다. 노트북의 `sys.executable`도 같은 `.venv/bin/python`을 가리켜야 합니다. 이 점검은 Google 로그인이나 원격 자원 할당을 하지 않습니다.

## Codespaces 설치·노트북

| 증상 | 확인과 해결 |
|---|---|
| 컨테이너 설정이 적용되지 않음 | `.devcontainer`가 저장소 최상위에 있는지 확인합니다. 이 프로젝트 폴더의 내용 전체를 독립 저장소 최상위에 두고 새 Codespace를 만듭니다. |
| `uv`를 찾을 수 없음 | 제공된 Dev Container를 다시 빌드합니다. 설치 스크립트는 컨테이너의 uv를 사용합니다. |
| 라이브러리 설치가 중단됨 | 네트워크 문제를 해결한 뒤 `bash scripts/setup.sh`를 다시 실행합니다. 라이브러리는 `requirements/codespaces.txt`에서 확인합니다. |
| `numpy`, `PIL`, `jax` 등을 찾지 못함 | 커널을 **Vision AI (Codespaces CPU)**로 선택합니다. 터미널의 Python과 노트북의 `sys.executable`을 비교하고, 설치 후 커널을 다시 시작합니다. `PIL`의 설치 이름은 `Pillow`입니다. |
| 데이터 파일 없음 | 00번 노트북을 위에서부터 실행해 다운로드·전처리·분할 저장을 마칩니다. 설정 스크립트는 데이터를 자동으로 준비하지 않습니다. 이후 `python scripts/doctor.py --data`로 검사합니다. |
| 데이터 준비가 중단됨 | 00번 노트북에서 실패한 단계와 다운로드 경로를 확인합니다. 원본 파일이 불완전하면 해당 캐시를 다시 받고 위에서부터 실행합니다. 기존 완성 데이터는 새 출력 폴더로 보존합니다. |
| 체크섬 오류 | 오류에 나온 파일과 출처를 확인합니다. 손상된 입력을 다시 준비하고 검사하며, 검증 코드를 꺼서 진행하지 않습니다. |
| GPU·TPU를 찾을 수 없다는 오류 | 02번·03번을 Codespaces CPU에서 실행했는지 확인합니다. [직접 실행 안내](direct-colab-cli.md)대로 Colab 세션을 준비하고 같은 노트북을 `colab exec -f`로 실행합니다. |
| `KernelClient`를 찾지 못함 | CLI 0.6.0에는 `jupyter-kernel-client==0.9.0`이 필요합니다. `bash scripts/setup.sh`로 명시된 버전을 설치하고 다시 점검합니다. |

## Google 로그인

```bash
source .venv/bin/activate
colab --auth oauth2 sessions
```

공식 CLI 0.6.0에는 `login` 서브커맨드가 없습니다. 위 세션 조회가 필요한 경우 로그인을 안내합니다. 안내된 주소를 브라우저에서 열고 받은 코드를 **터미널**에 입력하세요. 인증 코드를 노트북 셀이나 질문 게시판에 남기지 않습니다. 로그인은 코드가 표준 입력으로 전달되는 명령보다 먼저 마칩니다.

기본 방식은 `oauth2`입니다. 이미 Colab 범위의 Application Default Credentials를 준비한 경우에만 `colab --auth adc ...`를 사용하며 같은 세션 작업에 같은 인증 방식을 유지합니다. `adc` 옵션만 붙인다고 인증이 준비되지는 않습니다.

환경 설정은 가능한 경우 Colab CLI 상태를 워크스페이스의 `.local-state`에 연결합니다. 기존 인증 폴더는 덮어쓰지 않고 경고하므로 `doctor` 출력을 확인하세요. Codespaces를 삭제하면 워크스페이스의 인증·결과도 사라질 수 있습니다.

## GPU·TPU가 할당되지 않음

CU 잔액, 계정 제한, 해당 하드웨어의 가용성을 확인합니다. 기본 요청은 GPU T4, TPU v5e1입니다. 장치를 확인할 수 없으면 CPU로 바꾸어 실습 완료로 기록하지 않습니다.

GPU를 마쳤지만 TPU를 아직 실행하지 못했다면 `results/gpu`를 보관하고 나중에 새 TPU 세션에서 같은 데이터·설정으로 실행합니다. [직접 실행 안내](direct-colab-cli.md)의 TPU 단계를 따릅니다. 그전에는 TPU 항목을 미완료로 남깁니다.

## 브라우저를 닫았거나 실행이 중단됨

Codespaces와 Colab 세션은 별개입니다. Codespaces가 중지됐다고 Colab까지 종료됐다고 가정하지 않습니다.

```bash
colab sessions
colab status -s vision-gpu
```

`vision-gpu`는 문서의 예시 이름입니다. 다른 이름을 썼다면 실제 이름으로 바꿉니다. 출력 노트북과 원격 결과가 남았는지 확인하고 회수할 파일을 내려받습니다. CLI 호출 하나를 다시 실행하면 모든 단계가 자동으로 복구되는 구조는 아닙니다.

다시 학습하려면 이전 결과를 보관하고 이번 실습에서 만든 세션을 종료한 뒤 새 세션에서 준비부터 시작합니다. 살아 있는 학습 작업에 같은 노트북을 중복 실행하지 않습니다. 종료 명령은 다음과 같습니다.

```bash
colab stop -s vision-gpu
colab sessions
```

TPU를 사용했다면 실제 TPU 세션 이름으로도 확인합니다. 다른 실습의 세션은 종료하지 않습니다. CLI 상태가 사라졌다면 Colab의 런타임 관리 화면에서도 사용 중인 세션을 확인할 수 있습니다.

## 파일 전송이나 결과 회수가 실패함

`upload`는 파일 하나만 보내며 상위 폴더를 만들지 않습니다. [직접 실행 안내](direct-colab-cli.md)의 폴더 준비 셀을 먼저 실행했는지 확인합니다. requirements 설치만으로 모델과 데이터까지 전송되지는 않습니다.

다운로드는 로컬 상위 폴더를 먼저 만든 뒤 파일별로 재시도합니다.

```bash
mkdir -p results/gpu
colab download -s vision-gpu content/vision-ai/results/gpu/report.json results/gpu/report.json
```

남은 체크포인트와 CSV도 직접 실행 안내에 따라 회수합니다. 파일이 일부만 내려오면 완료가 아닙니다. 다운로드에 실패했다고 새 GPU를 반복해서 만들지 않습니다. 회수를 재시도해도 해결되지 않아 실습을 중단한다면 이번 실습의 세션을 종료하고, 회수하지 못한 파일과 실패 상황을 기록하세요. 종료하면 원격의 미회수 파일은 잃을 수 있습니다.

## CLI가 끝났는데 결과가 이상함

CLI 0.6.0은 원격 셀 오류를 출력한 뒤 다음 셀로 진행할 수 있습니다. `notebooks/02_gpu_finetuning_output.ipynb` 또는 `03_tpu_and_compare_output.ipynb`의 오류 출력부터 확인합니다. 보고서의 `status=completed`와 요청한 `device.platform`도 함께 검사합니다. 이전 실험의 보고서를 이번 실험 결과로 사용하지 않습니다.

`--timeout 1800`은 셀마다 실행을 기다리는 시간입니다. Colab 세션의 자동 종료 시간은 아니므로 실행·다운로드 오류가 있어도 이번 실습 세션의 `stop`, `sessions` 확인은 수행합니다.

## 메모리 부족·이미 완료된 결과 오류

02번·03번의 출력 노트북에서 실패한 셀을 찾습니다. 메모리 문제라면 학습 설정 셀의 `BATCH_SIZE`를 줄이고 GPU·TPU에 같은 값을 적용합니다. 이전 결과를 보관한 뒤 새 세션에서 실행합니다.

`완료된 결과가 있습니다` 오류는 `OUTPUT_DIR`에 기존 `report.json`이 있기 때문입니다. 그 결과를 보존하고 새 폴더를 지정하거나 새 세션에서 시작합니다. `OUTPUT_DIR`을 바꾸면 다운로드할 원격 경로도 바꿔야 합니다. 재실험 전 로컬 결과와 `*_output.ipynb`도 다른 폴더에 보관합니다.

## 파인튜닝 정확도가 오히려 낮음

버그라고 단정하기 전에 검증 손실, 혼동행렬, 클래스별 표본과 사진을 확인합니다. 작은 데이터에서 마지막 블록까지 바꾸면 과적합하거나 유용한 표현이 바뀔 수 있습니다. 학습률·epoch·데이터는 검증 결과로 조정하고 테스트는 최종 비교에 사용합니다. 결과가 나빠졌다는 사실도 보고서에 남깁니다.

## 직접 촬영한 이미지 준비·추론이 실행되지 않음

04번에서 `CUSTOM_INPUT`과 `PREPARE_CUSTOM=True`를 확인합니다. `train`, `validation`, `test`마다 같은 클래스 폴더가 있어야 하고 클래스는 2개 이상이어야 합니다. 지원 형식은 JPG·JPEG·PNG·WEBP입니다. 기존 출력 폴더를 덮어쓰려 하면 중단하므로 새 `CUSTOM_DATA`를 지정합니다.

추론은 `IMAGE_PATH`와 `CHECKPOINT`의 파일이 둘 다 있을 때 실행됩니다. 사진을 학습에 쓰려면 04번에서 저장한 네 데이터 파일을 직접 Colab으로 올리고 02번·03번을 실행합니다. 로컬 파일을 바꿨다고 원격 파일이 함께 바뀌지는 않습니다. 체크포인트의 클래스 이름과 순서도 확인하세요.

## 결과 파일은 어디에 있나

학습 결과는 `results/gpu`, `results/tpu`에 직접 내려받습니다. 실행 기록은 `notebooks/*_output.ipynb`에 있습니다. 이들은 아직 Codespaces의 파일이므로 VS Code 탐색기에서 내 컴퓨터로 내려받아 보관합니다. 인증 파일은 결과에 포함하지 않습니다.

Codespaces의 컴퓨팅·스토리지 비용과 Colab CU는 별도로 관리합니다. [GitHub 공식 비용 문서](https://docs.github.com/en/billing/concepts/product-billing/github-codespaces), [Colab FAQ](https://research.google.com/colaboratory/faq.html)
