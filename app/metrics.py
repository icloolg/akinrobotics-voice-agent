"""Per-turn latency measurement.

Usage:
    t = TurnTimer()          # start = moment the user stopped speaking
    ...; t.mark("stt_done")
    ...; t.mark("llm_first_token")
    t.log()
"""
import json
import logging
import time
from pathlib import Path

log = logging.getLogger("metrics")

TURNS_FILE = Path("logs/turns.jsonl")  # one JSON line per turn, read by scripts/benchmark.py


class TurnTimer:
    def __init__(self, turn_id: int = 0):
        self.turn_id = turn_id
        self.start = time.perf_counter()
        self.marks: dict[str, float] = {}
        self.extra: dict = {}

    def mark(self, name: str) -> None:
        """Record ms since start. Only the first call per name counts."""
        if name not in self.marks:
            self.marks[name] = round((time.perf_counter() - self.start) * 1000, 1)

    def summary(self) -> dict:
        return {"turn": self.turn_id, "ms": self.marks, **self.extra}

    def log(self) -> None:
        line = json.dumps(self.summary(), ensure_ascii=False)
        log.info("TURN %s", line)
        TURNS_FILE.parent.mkdir(exist_ok=True)
        with TURNS_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def rtf(processing_s: float, audio_s: float) -> float:
    """Real-Time Factor: < 1 means faster than real time."""
    return round(processing_s / audio_s, 3) if audio_s > 0 else 0.0
