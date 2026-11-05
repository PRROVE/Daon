import difflib
import re

from kiwipiepy import Kiwi

PRONOUN = re.compile(r"^(저|나|제|내|저희|저희들|우리)(는|은|가|이|도|의|하고는|하고|한테|에게|더러|보고|랑)?$")
APPEND_YO = {"어", "아", "여", "지", "네", "잖아", "거든", "더라", "더라고", "군", "는데", "ᆫ데", "은데", "데", "나", "ᆫ가", "는가"}
PLAIN_END = {"습니다", "ᆸ니다", "습니까", "ᆸ니까", "다", "ᆫ다", "는다", "니더", "십니까"}
SUBJECT = {"는": "은", "은": "은", "도": "도", "가": "이", "이": "이"}
OBLIQUE = {"한테", "에게", "더러", "보고", "랑"}


class StoryFormatter:
    """LLM 사연의 도입·마무리를 고정하고 1인칭·문장 끝을 규칙으로 맞춘다."""

    def __init__(self, intro: str, closing: str, max_chars: int) -> None:
        self.kiwi = Kiwi()
        self.intro = intro
        self.closing = closing
        self.max_chars = max_chars

    def format(self, story: str, transcript: str, nickname: str) -> str:
        """본문 어휘를 추측·삭제하지 않고 인칭과 문장 끝, 도입·마무리를 맞춘다."""
        # transcript와 max_chars는 호출 호환성을 유지한다. 내용 판정과 길이 검사는 평가 단계가 맡는다.
        body = self._sentences(story.strip())
        if body and self._frame_match(body[0], self.intro):
            body = body[1:]
        while body and self._frame_match(body[-1], self.closing):
            body = body[:-1]
        if not body:
            raise ValueError("LLM story has no body")
        lines = [self._sentence(sent, nickname) for sent in body]
        return " ".join([self.intro.format(nickname=nickname), *lines, self.closing.format(nickname=nickname)])

    @staticmethod
    def _frame_match(sentence: str, template: str) -> bool:
        # 별명 오타는 허용하되, 이야기·공감 같은 낱말만으로 본문을 지우지 않는다.
        pattern = re.escape(template).replace(re.escape("{nickname}"), r"[^.!?]+?")
        return re.fullmatch(pattern, sentence) is not None

    def _sentences(self, text: str) -> list[str]:
        # 형태소 기반 분리는 '좋다 ... 이런 생각은 없다'의 부정 범위를 끊을 수 있다.
        quoted = self._quotes(text)
        cuts = [0] + [m.end() for m in re.finditer(r"[.!?]+(?=\s|$)", text)
                      if not any(a <= m.start() < b for a, b in quoted)]
        return [text[a:b].strip() for a, b in zip(cuts, cuts[1:] + [len(text)]) if text[a:b].strip()]

    @staticmethod
    def _quotes(text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r'''"[^"]*"|'[^']*'|“[^”]*”|‘[^’]*’''', text)]

    def _sentence(self, sent: str, nickname: str) -> str:
        src = self._normalize(sent)
        toks = self.kiwi.tokenize(src)
        spans = self._person(toks, nickname, src)
        tail = self._tail(src, toks)
        if tail and not any(a <= tail[0] < b for a, b in self._quotes(src)):
            spans.append(tail)
        new = src
        for a, b, r in sorted(spans, key=lambda x: (x[0], x[1]), reverse=True):
            new = new[:a] + r + new[b:]
        q = new.rstrip().endswith("?") or bool(re.search(r"(습니까|ᆸ니까|십니까|니까)[.?! ]*$", src))
        new = new.rstrip(".?! ") + ("?" if q else ".")
        return new if self._safe(src.rstrip(".?! ") + (".?"[q]), new, nickname) else src

    @staticmethod
    def _normalize(s: str) -> str:
        # 방언 어미와 LLM 이 만든 겹친 어미는 형태소 분석이 깨지므로 분석 전에 표준형으로 돌린다
        s = re.sub(r"(어|아|해) ?가지고([.?! ]*)$", r"\1요\2", s)
        s = re.sub(r"(습|십)니더([.?! ]*)$", r"습니다\2", s)
        s = re.sub(r"지예([.?! ]*)$", r"지요\1", s)
        s = re.sub(r"(니다|다)요([.?! ]*)$", r"\1\2", s)
        s = re.sub(r"시요([.?! ]*)$", r"세요\1", s)
        return re.sub(r"같애([.?! ]*)$", r"같아\1", s)

    def _tail(self, s: str, toks: list) -> tuple[int, int, str] | None:
        q, i = s.rstrip().endswith("?"), len(toks) - 1
        while i >= 0 and toks[i].tag.startswith("S"):
            i -= 1
        if i < 1:
            return None
        if toks[i].form.endswith("요"):
            return self._broken(toks, i)
        e, p = toks[i], toks[i - 1]
        end = e.start + e.len
        if p.tag == "VCP":
            return self._copula(toks, i, end)
        if e.tag in ("EF", "EC") and e.form in APPEND_YO:
            return end, end, "요"
        verb = p.tag[:2] in ("VV", "VA", "VX", "XS", "EP")
        if verb and (e.form in PLAIN_END or (e.tag == "EC" and e.form == "고") or q):
            # 했·였은 어간과 시제가 한 글자로 합쳐져 있어 시제만 다시 조립하면 앞 글자가 사라진다
            if p.tag == "EP" and p.form != "시":
                return e.start, end, "어요"
            return p.start, end, self.kiwi.join([(p.form, p.tag), ("어요", "EF")])
        return None

    def _broken(self, toks: list, i: int) -> tuple[int, int, str] | None:
        # LLM 이 어간에 요를 바로 붙이거나(좋아하요) 어미 뒤에 예요를 덧붙인(입어예요) 경우만 고친다
        e, p = toks[i], toks[i - 1]
        end = e.start + e.len
        if e.form == "요" and p.tag[:2] in ("VV", "VA"):
            return p.start, end, self.kiwi.join([(p.form, p.tag), ("어요", "EF")])
        if i >= 2 and p.tag == "VCP" and toks[i - 2].tag == "EC":
            return toks[i - 2].start + toks[i - 2].len, end, "요"
        return None

    @staticmethod
    def _copula(toks: list, i: int, end: int) -> tuple[int, int, str] | None:
        prev = toks[i - 2] if i >= 2 else None
        # 조사·어미 뒤의 입니다는 예요로 바꾸면 불교로예요처럼 깨진다
        if prev is None or not prev.tag.startswith(("N", "XSN", "ETN")) or not "가" <= prev.form[-1] <= "힣":
            return None
        has_final = (ord(prev.form[-1]) - 0xAC00) % 28
        return prev.start, end, prev.form + ("이에요" if has_final else "예요")

    def _person(self, toks: list, nick: str, text: str) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        skip = -1
        quoted = self._quotes(text)
        for i, x in enumerate(toks):
            if i <= skip or any(a <= x.start < b for a, b in quoted):
                continue
            nx = toks[i + 1] if i + 1 < len(toks) else None
            if nx is not None:
                span, used = self._dialect_subject(toks, i, nick, text) or self._single(x, nx, nick, text)
                if span:
                    out.append(span)
                    skip = i + used
        return out

    @staticmethod
    def _dialect_subject(toks: list, i: int, nick: str, text: str) -> tuple[tuple[int, int, str], int] | None:
        # '내 조금 덜 먹고'의 내는 주어다. 관형절 '내 조금 남은 밥'은 소유 표현으로 둔다.
        x, nx = toks[i], toks[i + 1]
        if x.form != "나" or nx.tag != "JKG" or text[x.start:x.start + x.len] != "내":
            return None
        j = i + 2
        if j >= len(toks) or toks[j].tag != "MAG":
            return None
        while j < len(toks) and toks[j].tag == "MAG":
            j += 1
        if j >= len(toks) or toks[j].tag[:2] not in ("VV", "VA", "VX"):
            return None
        j += 1
        while j < len(toks) and toks[j].tag == "EP":
            j += 1
        if j < len(toks) and toks[j].tag in ("EC", "EF"):
            return (x.start, nx.start + nx.len, nick + "이"), 1
        return None

    def _single(self, x, nx, nick: str, text: str) -> tuple[tuple[int, int, str] | None, int]:
        after_det = text[:x.start].split()[-1:] in (["이"], ["그"], ["저"])
        if x.form == "저" and x.tag != "MM" and nx.form == "같":
            return (x.start, x.start + x.len, nick), 0
        if x.form in ("저", "나") and x.tag == "NP":
            if nx.tag in ("JX", "JKS") and nx.form in SUBJECT:
                return (x.start, nx.start + nx.len, nick + SUBJECT[nx.form]), 1
            if nx.tag == "JKG" and not after_det:
                return (x.start, nx.start + nx.len, nick + "의"), 1
            if nx.form in OBLIQUE or nx.tag.startswith("V"):
                return (x.start, x.start + x.len, nick), 0
            return None, 0
        if x.tag == "MM" and x.form in ("내", "제") and not after_det:
            if nx.tag == "MM" and nx.form == x.form:
                return (x.start, nx.start + nx.len, nick + "의"), 1
            if nx.tag.startswith("NN") or nx.tag == "MM":
                return (x.start, x.start + x.len, nick + "의"), 0
        return None, 0

    @staticmethod
    def _safe(old: str, new: str, nick: str) -> bool:
        # 인칭과 문장 끝 말고 다른 곳이 바뀌면 분석 오류라 변환을 포기한다
        a, b = old.split(), new.split()
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
            if op == "equal" or (i2 == len(a) and j2 == len(b)):
                continue
            pron = all(PRONOUN.match(w.rstrip(".,?!")) or w in ("내", "제") for w in a[i1:i2])
            if not (op != "delete" and pron and nick.split()[0] in " ".join(b[j1:j2])):
                return False
        return True


if __name__ == "__main__":
    f = StoryFormatter("오늘의 사연 주인공은 {nickname}이에요.", "비슷한 경험이 있으신 분은 {nickname}께 이야기를 들려주세요.", 308)
    transcript = "저는 고기를 좋아합니다 봄이면 꽃구경 가을이면 낙엽 지는 데를 갑니다 미신 종류는 굉장히 많더라고"
    print(f.format("도입입니다. 저는 고기를 좋아합니다. 봄이면 꽃구이나 가을이면 낙엽 지는 데를 갑니다. 미신 종류는 굉장히 많더라고. 참 공감이 돼요.", transcript, "은행 님"))
