---
name: vision-pipeline-inference
description: 이 저장소의 Hugging Face Pipeline 이미지 추론을 셸 파일로 Colab T4에서 실행하고 결과를 회수·검사·종료한다. 실행 계획이나 설명만 요청하면 dry-run으로 확인한다. 01·02 노트북의 AI 코딩 문제를 대신 풀거나 파인튜닝을 실행하는 스킬은 아니다.
---

# Vision Pipeline 추론 시연

`.vision-lab-root`가 있는 저장소 최상위에서 작업한다. [시연 안내](../../../hf_colab_gpu/pipeline-guide.md)와 `hf_colab_gpu/run_pipeline_colab.sh --help`를 읽어 입력·옵션·출력 경로를 확인한다. 추론 코드는 `hf_colab_gpu/pipeline_inference.py`이며 공개 `transformers.pipeline()`을 사용한다.

1. 설명·계획·검토·문서 수정 요청이면 `bash hf_colab_gpu/run_pipeline_colab.sh --dry-run`까지만 사용한다. 실제 원격 추론 요청이 있을 때 실행한다. 이미 허용된 실행을 거듭 확인하지 않는다.
2. 기본 입력은 00번에서 준비한 `data/prepared/test.npz`의 첫 cup 이미지다. 파일이 없으면 00번 준비 절차를 안내하거나 사용자가 지정한 이미지 경로를 사용한다. 다른 개인 파일을 임의로 검색·업로드하지 않는다. 사용자 이미지는 `--image`로 넘기며 셸 인자는 안전하게 인용한다.
3. 필요한 `.venv`와 CLI가 준비됐는지 확인하고 `source .venv/bin/activate` 후 `python scripts/doctor.py`를 실행한다. 실제 시연 전 `colab --auth oauth2 sessions`를 직접 실행해 인증을 확인한다. 첫 로그인은 사용자가 터미널 안내를 보고 마친다. 래퍼 내부 로그에는 인증 안내가 가려질 수 있으므로 첫 로그인을 래퍼에 맡기지 않는다. 기존 ADC 인증을 쓰기로 한 경우 `colab --auth adc sessions`와 래퍼의 `--auth adc`를 일관되게 사용한다. 인증 코드·토큰을 문서나 커밋에 남기지 않는다.
4. `bash hf_colab_gpu/run_pipeline_colab.sh`를 실행한다. 별도 지정이 없다면 자동 생성된 새 세션 이름과 결과 폴더를 쓴다. T4 가용량이 없다고 다른 장치나 새 자원을 반복 할당하지 않는다. 이 래퍼는 GPU가 없으면 CPU로 대체하지 않으며, 기존 세션도 재사용하지 않는다.
5. 스크립트의 종료코드, 이번 실행의 JSON과 실행 ID, 실제 GPU, 모델·revision, 예측 목록, 세션 종료 확인을 함께 검사한다. 오류가 발생하면 실패 단계와 남은 결과·로그를 보고한다. 래퍼는 결과 다운로드 실패를 포함해 종료 시 이번 세션을 정리하므로 종료된 런타임의 파일을 회수했다고 주장하지 않는다. 종료 확인이 실패하면 이 실행이 만든 세션만 다시 확인·정리한다.
6. 사용자에게 결과 파일 위치, 상위 예측, 실행 장치, 세션 종료 상태를 짧게 전달한다. 원본 모델은 ImageNet 1,000개 라벨을 출력하며 예측 점수를 정확도라고 부르지 않는다. 과거 검증 기록을 이번 실행 결과로 사용하지 않는다.

이 스킬은 추론 시연을 실행한다. 학생의 네 빈칸을 자동으로 채우거나 별도 정답 노트북을 만들지 않는다. 00·01·02번 전체 학습 흐름은 [기본 실습 안내](../../../hf_colab_gpu/README.md)를 따른다.
