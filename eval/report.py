import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from config.settings import Settings
from eval.format_check import FormatChecker

RULES = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")
# (표시 이름, summary 안의 경로, thresholds 키, 합격 방향)
KEY_METRICS = (
    ("① 보존율(macro)", ("content", "fact_recall_macro"), "fact_recall_macro", ">="),
    ("① 왜곡·추가 발생률", ("content", "inconsistency_rate"), "inconsistency_rate", "<="),
    ("② 형식 전체 충족률", ("format", "format_rate"), "format_rate", ">="),
    ("③ E2E 성공률", ("stability", "e2e_success_rate"), "e2e_success_rate", ">="),
    ("④ 통합 통과율", ("overall", "pass_rate"), "overall_pass_rate", ">="),
)


class Report:
    """pipeline/judgments 를 합쳐 요청별 records.jsonl 과 ①~④ summary.json 을 만들고 표로 출력한다."""

    def __init__(self, checker: FormatChecker, e2e_timeout_sec: float, thresholds: dict) -> None:
        self.checker = checker
        self.e2e_timeout_sec = e2e_timeout_sec
        self.thresholds = thresholds

    def summarize(self, run_dir: Path) -> dict:
        """최종 레코드와 집계를 저장하고 집계를 돌려준다. 같은 세트의 이전 run 이 있으면 나란히 비교한다."""
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        judg = self._latest_by_id(run_dir / "judgments.jsonl")
        records = [self._merge(r, judg.get(r["request_id"]))
                   for r in self._read(run_dir / "pipeline.jsonl") if not r["warmup"]]
        with (run_dir / "records.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {
            "run_id": run["run_id"], "set_name": run["set_name"], "n_samples": run["n_samples"],
            "n_requests": len(records), "warmup_excluded": run["warmup"], "env": run["env"],
            "story_prompt_sha256": run["prompt_sha256"], "judge": self._judge_info(judg),
            "content": self._content(records), "format": self._format(records),
            "stability": self._stability(records), "overall": self._overall(records),
            "thresholds": self.thresholds,
        }
        summary["verdict"] = self._verdict(summary)
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._print(summary)
        self._compare(run_dir, summary)
        return summary

    def _merge(self, rec: dict, j: dict | None) -> dict:
        chk = self.checker.check(rec["story"], rec["out_wav"], rec.get("ref_transcript", ""))
        j = self._validated_judgment(rec, j)
        judged = j is not None and j["error"] is None
        llm = j if judged and j["generated"] else None
        rules = self._rules(chk, llm)
        out = {
            **rec, "generated": bool((rec["story"] or "").strip()), "judged": judged,
            "judge_error": None if j is None else j["error"], **self._content_fields(j, judged),
            "rules": rules, "format_pass": None if None in rules.values() else all(rules.values()),
            "chars": chk["R2"]["chars"], "r5_regex_hits": chk["R5_regex"]["hits"],
            "audio_valid": chk["audio"]["valid"], "audio_reason": chk["audio"]["reason"],
            "output_duration": chk["audio"]["duration"],
            "e2e_success": bool(rec["ok"] and chk["audio"]["valid"] and rec["e2e_sec"] <= self.e2e_timeout_sec),
            "judge_detail": None if llm is None else {"content": llm["content"], "format": llm["format"]},
        }
        out["overall_pass"] = self._overall_pass(out)
        return out

    @staticmethod
    def _content_fields(j: dict | None, judged: bool) -> dict:
        empty = {"fact_recall": None, "inconsistent": None, "distortion": None, "addition": None,
                 "preservation_method": None}
        if j is None:
            return empty
        if not j["generated"]:
            # 평가지표 ①: 생성 실패(빈 출력)는 보존 0, 왜곡·추가 있음으로 집계
            return {"fact_recall": 0.0, "inconsistent": True, "distortion": True, "addition": True,
                    "preservation_method": "generation_failed"}
        if not judged:
            return empty
        c = j["content"]
        ratio, method = c["preservation_ratio"], "laya_estimate"
        return {"fact_recall": ratio, "preservation_method": method,
                "distortion": bool(c["distortion"]["found"]), "addition": bool(c["addition"]["found"]),
                "inconsistent": bool(c["distortion"]["found"] or c["addition"]["found"])}

    @staticmethod
    def _validated_judgment(rec: dict, j: dict | None) -> dict | None:
        if j is None:
            return j
        if j.get("schema") != "laya-v1":
            raise ValueError("과거 판정 형식은 재집계하지 않습니다. 기존 summary.json을 확인하세요.")
        state = {"nickname": rec.get("nickname"), "reference": rec.get("ref_transcript"), "story": rec.get("story")}
        digest = hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest != j.get("input_sha256"):
            return {**j, "error": "판정 이후 입력이 변경됨: eval.judge 재실행 필요"}
        return j

    @classmethod
    def _rules(cls, chk: dict, llm: dict | None) -> dict:
        f = (llm or {}).get("format") or {}
        # R5 는 정규식과 LLM 판정을 병행: 둘 다 통과해야 통과
        r5 = None if cls._llm_pass(f, "R5") is None else bool(chk["R5_regex"]["pass"] and cls._llm_pass(f, "R5"))
        return {"R1": cls._llm_pass(f, "R1"), "R2": chk["R2"]["pass"], "R3": cls._llm_pass(f, "R3"),
                "R4": cls._llm_pass(f, "R4"), "R5": r5, "R6": chk["R6"]["pass"], "R7": cls._llm_pass(f, "R7")}

    @staticmethod
    def _llm_pass(f: dict, key: str) -> bool | None:
        return None if key not in f else bool(f[key]["pass"])

    def _overall_pass(self, r: dict) -> bool:
        # 평가지표 ④: 생성 실패·시간초과(E2E 실패)·미평가는 모두 미통과
        return bool(r["e2e_success"] and r["generated"] and r["judged"]
                    and r["fact_recall"] == 1.0 and not r["inconsistent"] and r["format_pass"])

    @staticmethod
    def _rate(num: int, den: int) -> dict:
        # 분모 0 은 0% 가 아니라 "계산 불가"
        return {"value": round(100 * num / den, 2) if den else None, "numerator": num, "denominator": den,
                "note": None if den else "계산 불가"}

    def _content(self, records: list[dict]) -> dict:
        judged = [r for r in records if r["judged"]]
        recalls = [r["fact_recall"] for r in judged if r["fact_recall"] is not None]
        unjudged = [r["request_id"] for r in records if not r["judged"]]
        return {
            # 합격 판정은 사연별 평균(macro). 설정된 반복 실행 중 판정된 요청을 모두 포함한다.
            "fact_recall_macro": {"value": round(100 * statistics.mean(recalls), 2) if recalls else None,
                                  "numerator": "사연별 보존율 평균", "denominator": len(recalls),
                                  "note": None if recalls else "계산 불가"},
            "preservation_methods": dict(Counter(r["preservation_method"] for r in judged)),
            "preservation_note": "Laya 점수 평균은 추정 보존율이며 사실별 정답 개수를 세는 지표와 다름",
            "distortion_rate": self._rate(sum(bool(r["distortion"]) for r in judged), len(judged)),
            "addition_rate": self._rate(sum(bool(r["addition"]) for r in judged), len(judged)),
            "inconsistency_rate": self._rate(sum(bool(r["inconsistent"]) for r in judged), len(judged)),
            "unjudged": {"count": len(unjudged), "request_ids": unjudged},
            "provisional": bool(unjudged),
        }

    def _format(self, records: list[dict]) -> dict:
        generated = [r for r in records if r["generated"]]
        decided = [r for r in generated if r["format_pass"] is not None]
        valid = [r for r in records if r["audio_valid"]]
        return {
            # 분모 = 생성된 사연(빈 출력 제외) 중 판정이 끝난 것. 미평가는 따로 센다
            "format_rate": self._rate(sum(r["format_pass"] for r in decided), len(decided)),
            "format_unjudged": len(generated) - len(decided),
            "rule_fail_counts": {k: sum(1 for r in decided if r["rules"][k] is False) for k in RULES},
            "invalid_audio": {"count": len(records) - len(valid),
                              "request_ids": [r["request_id"] for r in records if not r["audio_valid"]]},
        }

    def _stability(self, records: list[dict]) -> dict:
        succ = [r for r in records if r["e2e_success"]]
        out = {
            "e2e_success_rate": self._rate(len(succ), len(records)),
            "failures": {"by_stage": dict(Counter(r["failed_stage"] for r in records if not r["ok"])),
                         "timeout": sum(1 for r in records if r["ok"] and r["audio_valid"] and r["e2e_sec"] > self.e2e_timeout_sec),
                         "invalid_audio": sum(1 for r in records if r["ok"] and not r["audio_valid"])},
            "dataset": "AI Hub 중·노년층 방언 1인발화 질문에답하기 (단일 데이터셋)", "by_length_bucket": {},
        }
        for b in sorted({r["length_bucket"] for r in records}):
            rs = [r for r in records if r["length_bucket"] == b]
            out["by_length_bucket"][b] = self._rate(sum(r["e2e_success"] for r in rs), len(rs))
        return out

    def _overall(self, records: list[dict]) -> dict:
        topics = sorted({r["topic"] for r in records})
        return {
            "pass_rate": self._rate(sum(r["overall_pass"] for r in records), len(records)),
            "note": "Laya 사용 시 보존율 점수 100%를 전부 보존의 추정 조건으로 사용. E2E는 실제 실행 기록.",
            "by_topic": {t: self._rate(sum(r["overall_pass"] for r in records if r["topic"] == t),
                                       sum(1 for r in records if r["topic"] == t)) for t in topics},
            "failed": [{"request_id": r["request_id"], "reasons": self._fail_reasons(r)} for r in records if not r["overall_pass"]],
        }

    def _fail_reasons(self, r: dict) -> list[str]:
        reasons = []
        if not r["ok"]:
            reasons.append(f"생성 실패@{r['failed_stage']}: {str(r['error'])[:60]}")
        elif not r["audio_valid"]:
            reasons.append(f"음성 비정상: {r['audio_reason']}")
        if r["ok"] and r["audio_valid"] and r["e2e_sec"] > self.e2e_timeout_sec:
            reasons.append(f"시간초과 {r['e2e_sec']:.1f}s > {self.e2e_timeout_sec}s")
        if r["generated"] and not r["judged"]:
            reasons.append("미평가" + (f": {r['judge_error'][:60]}" if r["judge_error"] else ""))
        if r["judge_detail"]:
            if r["fact_recall"] is not None and r["fact_recall"] < 1.0:
                reasons.append(f"보존율 {100*r['fact_recall']:.2f}% < 100% ({r['preservation_method']})")
            if r["inconsistent"]:
                reasons.append("왜곡·추가")
        failed_rules = [k for k, v in r["rules"].items() if v is False]
        if r["generated"] and failed_rules:
            reasons.append("형식 " + ",".join(failed_rules))
        return reasons

    def _verdict(self, summary: dict) -> dict:
        out = {}
        for _, path, key, op in KEY_METRICS:
            v, t = self._value(summary, path), self.thresholds[key]
            out[key] = None if v is None else ("PASS" if (v >= t if op == ">=" else v <= t) else "FAIL")
        return out

    @staticmethod
    def _value(summary: dict, path: tuple[str, str]) -> float | None:
        node = summary[path[0]][path[1]]
        return node["value"] if isinstance(node, dict) else node

    @staticmethod
    def _judge_info(judg: dict) -> dict:
        lines = list(judg.values())
        if not lines:
            return {"note": "판정 없음"}
        j = lines[0]
        return {"models": sorted({x["judge_model"] for x in lines}),
                "signatures": sorted({x.get("judge_signature", "historical") for x in lines}),
                "metadata": j.get("judge_metadata", {"model": j["judge_model"],
                                                     "prompt_sha256": j.get("prompt_sha256")})}

    def _print(self, s: dict) -> None:
        print(f"\n=== {s['run_id']} · {s['set_name']} · 샘플 {s['n_samples']} · 요청 {s['n_requests']} (워밍업 {s['warmup_excluded']}건 제외) ===")
        print(f"{'지표':<22}{'값':>10}  {'분자/분모':<22}{'합격선':<8}판정")
        for name, path, key, op in KEY_METRICS:
            node = s[path[0]][path[1]]
            vs = f"{node['value']:.1f}%" if node["value"] is not None else "계산 불가"
            print(f"{name:<22}{vs:>10}  {node['numerator']}/{node['denominator']:<20}{op}{self.thresholds[key]}%  {s['verdict'][key] or '-'}")
        c, f, st = s["content"], s["format"], s["stability"]
        print(f"① 왜곡 {self._fmt(c['distortion_rate'])} | 추가 {self._fmt(c['addition_rate'])} | 미평가 {c['unjudged']['count']}건{' → 잠정 결과' if c['provisional'] else ''}")
        print(f"① 보존율 방식 {c['preservation_methods']}")
        print(f"② 규칙별 실패 {f['rule_fail_counts']} | 형식 미평가 {f['format_unjudged']} | 음성 비정상 {f['invalid_audio']['count']}")
        print(f"③ 실패 {st['failures']} | 길이별 성공률 {{{', '.join(f'{b}: {self._fmt(v)}' for b, v in st['by_length_bucket'].items())}}}")
        print(f"④ 주제별 {{{', '.join(f'{t}: {self._fmt(v)}' for t, v in s['overall']['by_topic'].items())}}}")
        if s["overall"]["failed"]:
            print("\n통과 못 한 요청:")
            for item in s["overall"]["failed"]:
                print(f"  {item['request_id']:<10} " + " / ".join(item["reasons"]))

    def _compare(self, run_dir: Path, summary: dict) -> None:
        prev = self._previous_run(run_dir, summary["set_name"])
        if prev is None:
            print("\n같은 세트의 이전 run 없음 (비교 생략)")
            return
        ps = json.loads((prev / "summary.json").read_text(encoding="utf-8"))
        if ps.get("judge") != summary.get("judge"):
            print("\n평가자·가중치·판정 설정이 다른 이전 결과: 직접 증감 비교 생략")
            return
        print(f"\n=== 이전 run 비교: {ps['run_id']} → {summary['run_id']} ===")
        for name, path, _, _ in KEY_METRICS:
            a, b = self._value(ps, path), self._value(summary, path)
            diff = f"{b - a:+.1f}" if a is not None and b is not None else "-"
            print(f"{name:<22}{self._num(a):>10} → {self._num(b):<10} ({diff})")

    @staticmethod
    def _previous_run(run_dir: Path, set_name: str) -> Path | None:
        runs = sorted(p for p in run_dir.parent.glob(f"{set_name}_*") if p != run_dir and (p / "summary.json").exists())
        return runs[-1] if runs else None

    @staticmethod
    def _fmt(rate: dict) -> str:
        return "계산 불가" if rate["value"] is None else f"{rate['value']}% ({rate['numerator']}/{rate['denominator']})"

    @staticmethod
    def _num(v: float | None) -> str:
        return "계산 불가" if v is None else f"{v}"

    @staticmethod
    def _read(path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    @classmethod
    def _latest_by_id(cls, path: Path) -> dict[str, dict]:
        return {line["request_id"]: line for line in cls._read(path)}


if __name__ == "__main__":
    s = Settings()
    e = s.eval
    run_dir = e["results_dir"] / sys.argv[1]
    checker = FormatChecker(s.llm["target_chars"]["max"], e["intro"])
    Report(checker, e["e2e_timeout_sec"], e["thresholds"]).summarize(run_dir)
