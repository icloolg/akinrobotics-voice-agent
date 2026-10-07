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

log = logging.getLogger("metrics")


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
        # One JSON line per turn -> scripts/benchmark.py can parse the log file.
        log.info("TURN %s", json.dumps(self.summary(), ensure_ascii=False))


def rtf(processing_s: float, audio_s: float) -> float:
    """Real-Time Factor: < 1 means faster than real time."""
    return round(processing_s / audio_s, 3) if audio_s > 0 else 0.0
