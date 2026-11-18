# 데이터셋 생성·평가

[프로젝트 소개](../README.md) · [설치·모델 준비](SETUP.md) · [평가지표](평가지표.md)

데이터셋의 음성을 사연과 출력 음성으로 만든 뒤, 원본 정답 전사문을 기준으로 내용·형식을 판정하고 처리시간과 함께 집계합니다. [설치 안내](SETUP.md)에 따라 Laya를 포함한 모델 준비를 마친 후 진행합니다.

자기 WAV 한 건으로 사연만 만들려면 [첫 사연 생성](SETUP.md#5-첫-사연-생성)을 참고하세요.

## 데이터 준비

`data/{set_name}/manifest.jsonl`에 샘플별 정보를 한 줄씩 저장합니다. 현재 실행 가능한 세트는 `dev_clear`, `dev_set`, `eval_set`입니다.

필수 필드는 `sample_id`, `speaker_id`, `topic`, `nickname`, `length_bucket`, `episode_id`, `duration`, `wav`, `transcript`입니다. `wav`와 `transcript`는 해당 세트 폴더 기준 파일 경로입니다. [manifest 예제와 준비 안내](../examples/README.md)에서 작성 방법을 확인할 수 있습니다.

- **생성 입력:** `wav`의 음성을 STT로 전사한 텍스트.
- **평가 기준:** `transcript`가 가리키는 원본 라벨 정답 텍스트. 실행 시 `ref_transcript`로 자동 저장합니다.

## 생성 → 판정 → 집계

프로젝트 루트의 PowerShell에서 순서대로 실행합니다.

```powershell
# 1. 음성에서 사연과 출력 음성 생성
conda run -n daon_v2 python -m eval.run_pipeline dev_clear

# 2. 첫 단계에서 출력된 결과 폴더 이름 입력
$runId = "dev_clear_YYYYMMDD_HHMMSS"
conda run -n daon_v2 python -m eval.judge $runId

# 3. 코드 검사 및 최종 집계
conda run -n daon_v2 python -m eval.report $runId
```

판정 단계는 입력과 모델 설정이 같은 완료 건을 건너뛰고, 미평가·변경 건을 다시 판정합니다. 기존 사연의 재평가는 음성 생성이나 E2E 시간을 다시 측정하지 않습니다.

## 실행 설정

현재 [`config/config.yaml`](../config/config.yaml)의 기본값입니다.

| 설정 | 값 |
|---|---|
| STT 장치 | `device: cpu` |
| Qwen 장치 | `llm.num_gpu: 0` — CPU |
| Laya 장치 | `judge.device: auto` — CUDA 사용 가능 시 GPU |
| 실행 횟수 | 샘플당 1회, 워밍업 1건은 집계에서 제외 |
| E2E 제한 | 120초 |
| 사연 길이 상한 | 공백 포함 308자 |

E2E는 음성을 STT에 넘기는 시점부터 전체 TTS 음성 저장 완료까지 측정합니다. Laya 평가 시간은 포함하지 않습니다. 판정 규칙·산식·합격 기준은 [평가지표](평가지표.md)를 참고하세요.

## 결과 파일

결과는 `results/{run_id}/`에 저장합니다.

| 파일 | 내용 |
|---|---|
| `run.json`, `config.yaml` | 실행 환경·모델·설정 기록 |
| `pipeline.jsonl`, `audio/` | 전사문·사연·정답 원문·처리시간·출력 음성 |
| `judgments.jsonl` | Laya 점수·판정·오류 기록 |
| `records.jsonl` | 사연별 내용·형식·E2E·통합 판정 |
| `summary.json` | 보존율·왜곡률·추가율·형식 충족률·E2E 성공률·통합 통과율 |

입력 누락·잘림·추론 오류는 미평가로 기록하며, 미평가가 있으면 내용 집계는 잠정 결과로 표시합니다. Laya는 오류 구절에 대한 설명문을 생성하지 않습니다.

## 공개 평가 기록

[2026-11-05 공개 평가 결과](../README.md#평가-결과)에서 `dev_clear` 28건의 평가 조건과 상세 수치를 확인할 수 있습니다. 보존율은 모델 추정값이며, E2E는 원본 생성 실행의 시간과 음성 검사 결과를 재사용했습니다. 이 28건의 학습 데이터 중복 여부는 검증하지 않았습니다.

[결과 그래프](Daon_evaluation_28.png) · [집계 원본](../results/published_20261105/summary.json) · [현재 한계](ISSUES.md)
