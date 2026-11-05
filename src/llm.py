import re
from pathlib import Path

import requests

from config.settings import Settings


class LLM:
    """Ollama 의 LLM 으로 전사문을 라디오 사연으로 바꾼다."""

    def __init__(self, host: str, model: str, think: bool | None, num_gpu: int | None, timeout_sec: float, prompt_path: Path) -> None:
        self.host = host
        self.model = model
        self.think = think
        self.num_gpu = num_gpu
        self.timeout_sec = timeout_sec
        # 프롬프트 파일은 "## system" / "## user" 두 섹션. {nickname} {transcript} 자리를 채워 보낸다
        parts = re.split(r"^## (\w+)\s*$", prompt_path.read_text(encoding="utf-8"), flags=re.MULTILINE)
        self.sections = {k: v.strip() for k, v in zip(parts[1::2], parts[2::2])}

    def generate(self, transcript: str, nickname: str) -> str:
        """전사문과 별명으로 사연 본문을 생성해 돌려준다."""
        fields = {"nickname": nickname, "transcript": transcript}
        return self._chat(self.sections["system"].format(**fields), self.sections["user"].format(**fields))

    def _chat(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
        }
        if self.think is not None:
            body["think"] = self.think
        # num_gpu 0 = CPU 강제. null 이면 Ollama 기본 배치
        if self.num_gpu is not None:
            body["options"] = {"num_gpu": self.num_gpu}
        # Ollama 가 멈추면 run 전체가 무한 대기하므로 상한을 둔다. 넘으면 예외가 Pipeline 에 실패로 기록된다
        resp = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_sec)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


if __name__ == "__main__":
    cfg = Settings().llm
    llm = LLM(host=cfg["host"], model=cfg["model"], think=cfg["think"], num_gpu=cfg["num_gpu"],
              timeout_sec=cfg["timeout_sec"], prompt_path=cfg["prompt_path"])
    transcript = (
        "그래도 뭐 우리 어머니는 아침마다 공원에 가서 운동을 하셨어요. "
        "나도 요즘은 복지관에서 노래 교실을 다니는데 그게 참 재밌어요. "
        "일주일에 두 번 가는데 가면 친구들도 만나고 그래요."
    )
    story = llm.generate(transcript, nickname="바다 님")
    print(story)
    print(len(story), "chars")
