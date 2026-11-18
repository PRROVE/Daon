# 입력 데이터 예제

음성 한 건으로 사연만 만들 때는 데이터셋 없이 직접 실행할 수 있다.

```powershell
conda run --no-capture-output -n daon_v2 python scripts/demo_story.py --audio "C:\audio\recording.wav" --nickname "바다 님"
```

`--out-dir`로 레포지토리 밖의 저장 위치도 지정할 수 있다. `--wait`는 녹화할 때 실행 전후에 Enter 입력을 기다리는 옵션이다. 이 명령은 사연 생성용이며, Laya 내용 평가는 원본 정답 전사문이 있는 데이터셋으로 별도 실행한다.

## 데이터셋 평가용 manifest

`manifest.example.jsonl`은 형식 예제다. 실제 음성과 정답 전사문은 포함하지 않는다.

1. `data/dev_set/` 폴더를 만들고 예제 파일을 `manifest.jsonl`로 복사한다.
2. 직접 준비한 `example01.wav`와 그 음성을 정확히 옮긴 `example01.txt`를 같은 폴더에 둔다. 전사문은 UTF-8로 저장한다.
3. `duration`의 자리표시자 `0.0`을 실제 음성 길이(초)로 바꾸고, `length_bucket`에는 길이 구간의 상한을 적는다. 예를 들어 15초 음성은 `20`으로 분류한다.
4. 화자·주제·닉네임·에피소드 정보를 실제 데이터에 맞게 수정한다. 여러 건은 한 줄에 하나씩 추가하고 `sample_id`는 중복 없이 사용한다.

`wav`와 `transcript` 경로는 manifest가 있는 폴더 기준이다. 환경 준비는 [설치 안내](../docs/SETUP.md), 데이터셋 생성·판정·집계 명령은 [평가 실행 안내](../docs/EVALUATION.md)를 참고한다.
