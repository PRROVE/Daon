# 설치·모델 준비

[프로젝트 소개](../README.md) · [평가 실행](EVALUATION.md) · [구조·개발 안내](ARCHITECTURE.md)

확인 환경은 Windows, Python 3.11, conda 환경 `daon_v2`입니다. conda, Git, [Ollama](https://ollama.com/)를 준비한 뒤 진행합니다. ffmpeg는 아래 단계에서 설치합니다.

처음 내려받는 경우 다음 명령으로 시작합니다. 이후 모든 명령은 프로젝트 루트의 PowerShell에서 실행합니다.

```powershell
git clone https://github.com/PRROVE/Daon.git Daon_v2
cd Daon_v2
```

## 1. Python 환경·패키지 설치

```powershell
conda create -n daon_v2 python=3.11 -y
$env:PYTHONUTF8 = "1"
conda install -n daon_v2 -c conda-forge "ffmpeg=7.1.0=gpl_h2585aa8_705" -y
```

PyTorch는 실행 장치에 맞는 명령 하나를 선택합니다. 기존 검증 환경은 `2.11.0+cu128`이며, CUDA 사용에는 호환되는 NVIDIA 드라이버가 필요합니다.

```powershell
# NVIDIA GPU: 기존 검증 환경과 같은 CUDA 12.8 빌드
conda run -n daon_v2 pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# CPU만 사용하는 환경
conda run -n daon_v2 pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

나머지 패키지를 설치하고 의존성을 확인합니다.

```powershell
conda run -n daon_v2 pip install -r requirements.txt
conda run -n daon_v2 pip check
```

후처리에는 [Kiwi · kiwipiepy](https://github.com/bab2min/kiwipiepy)의 형태소 분석을 사용하며, `kiwipiepy==0.24.1`은 위 의존성에 포함되어 있습니다.

[`requirements.txt`](../requirements.txt)는 기존 환경에서 확인한 직접 의존성과 Laya 호환 패키지 버전입니다. 모든 전이 의존성을 고정한 환경 잠금 파일은 아니며, Jetson 설치·구동은 별도 검증이 필요합니다.

## 2. 사연 생성 모델 준비

### KoMix V2 — 음성 전사

[KoMix V2 릴리스](https://github.com/PRROVE/Daon/releases/tag/komixv2-20261105)에서 [komixv2.pt](https://github.com/PRROVE/Daon/releases/download/komixv2-20261105/komixv2.pt)를 받아 `models/whisper/komixv2.pt`에 저장합니다. 릴리스 본문에 설치 안내와 SHA256이 있습니다.

원본은 [seastar105/whisper-medium-komixv2](https://huggingface.co/seastar105/whisper-medium-komixv2)이며, 배포 파일은 [openai-whisper](https://github.com/openai/whisper) 형식으로 변환한 가중치입니다.

### Silero VAD — 긴 음성 분할

[Silero VAD v6.2.3](https://github.com/snakers4/silero-vad/releases/tag/v6.2.3)의 저장소와 모델을 `models/vad/silero-vad/`에 준비합니다.

```powershell
git clone --depth 1 --branch v6.2.3 https://github.com/snakers4/silero-vad.git models/vad/silero-vad
```

### Qwen · Ollama — 사연 생성

별도 터미널에서 모델 저장 위치를 `models/ollama/`로 설정하고 Ollama 서버를 실행합니다. 같은 포트에 실행 중인 서버가 있다면 모델 경로 설정을 확인합니다.

```powershell
$env:OLLAMA_MODELS = (Join-Path (Get-Location) "models/ollama")
ollama serve
```

서버가 실행된 상태에서 다른 터미널로 모델을 받습니다. [Qwen3-4B-Instruct-2507 원본](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)의 [Ollama Q4_K_M 배포본](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M)을 사용하며, 설정에 쓰는 이름으로 복사합니다.

```powershell
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama cp qwen3:4b-instruct-2507-q4_K_M qwen3:4b-instruct
```

### Supertonic — 음성 합성

`supertonic==1.3.1`의 기본 모델인 [Supertonic 3](https://huggingface.co/Supertone/supertonic-3)을 사용합니다. 최초 실행 시 `models/supertonic/`으로 자동 다운로드합니다. 구현은 [Supertonic 보관 저장소](https://github.com/supertone-oss-archive/supertonic)와 [Python SDK 보관 저장소](https://github.com/supertone-oss-archive/supertonic-py)를 참고하세요.

## 3. 평가 모델 준비

이 단계는 데이터셋의 내용·규칙 평가에 필요합니다. 자기 WAV 파일로 사연만 생성할 때는 생략할 수 있습니다.

[Laya 릴리스](https://github.com/PRROVE/Daon/releases/tag/daon-v2-20261105)에서 [laya-real400-repair001.zip](https://github.com/PRROVE/Daon/releases/download/daon-v2-20261105/laya-real400-repair001.zip)을 받아 프로젝트 루트에 압축을 해제합니다. 가중치·설정·토크나이저·`questions.json`이 `models/laya/`에 함께 있어야 합니다. 릴리스 본문에 설치 안내와 SHA256이 있습니다.

[Laya](https://github.com/NandhaKishorM/laya)와 기반 인코더 [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base)를 바탕으로 프로젝트 데이터로 학습한 `laya-real400-repair001`을 사용합니다.

## 4. 실행 환경 점검

모델 파일과 패키지 버전, ffmpeg, Ollama 서버·모델 등록 상태를 확인합니다. 점검 자체는 모델 추론이나 다운로드를 수행하지 않습니다.

```powershell
# 사연 생성만 사용하는 경우
conda run -n daon_v2 python scripts/check_env.py --generation-only

# Laya를 포함해 데이터셋 평가까지 사용하는 경우
conda run -n daon_v2 python scripts/check_env.py
```

사용 목적에 맞는 명령을 실행하고 `[FAIL]` 항목을 해결합니다. 실제 경로·장치 설정은 [`config/config.yaml`](../config/config.yaml)을 기준으로 합니다. Laya 가중치·질문 파일과 사연 프롬프트는 설정의 SHA256과 대조합니다.

## 5. 첫 사연 생성

직접 녹음한 WAV의 경로와 닉네임을 지정합니다.

```powershell
conda run --no-capture-output -n daon_v2 python scripts/demo_story.py --audio "C:\audio\recording.wav" --nickname "바다 님"
```

실행 기록과 사연 텍스트는 `results/demo_날짜_시간/`에, 출력 음성은 그 안의 `audio/`에 저장합니다. `--out-dir`로 저장 위치를 바꿀 수 있습니다. 이 명령은 사연 생성용이며 Laya 평가는 수행하지 않습니다.

README의 GIF와 같은 샘플은 로컬에 `data/dev_clear/manifest.jsonl`과 해당 음성이 준비된 경우 실행할 수 있습니다.

```powershell
conda run --no-capture-output -n daon_v2 python scripts/demo_story.py --sample c04
```

여러 건을 생성하고 평가하려면 [입력 데이터 준비](../examples/README.md)와 [평가 실행 안내](EVALUATION.md)를 참고하세요.
