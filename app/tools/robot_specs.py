"""Robot specifications database (SQLite) for comparison questions.

Why a database next to RAG (measured, see NOTLAR.md): RAG gives the LLM only
the 2 most relevant chunks, but "En hafif robot hangisi?" needs all 8 robots.
Before this tool, 5 comparison questions got 0 correct answers.

Why no text-to-SQL: a 3B model writes unreliable SQL, and putting text from
speech into SQL is a security risk. Instead the question selects one column
from a fixed whitelist, and the query is fixed:
    SELECT name, <column> FROM robots ORDER BY <column>
The LLM gets every robot sorted by that value, plus the min and max, and can
answer "en hafif", "en hızlı" from it. A number filter ("8 saatten fazla")
is done in SQL, not by the LLM: measured, the LLM listed an 8-hour robot as
"more than 8 hours".

The database is built from knowledge/robot_specs.csv by scripts/ingest.py.
"""
import csv
import re
import sqlite3
from pathlib import Path

from app.tools.base import Tool, ToolNotApplicable, ToolUnavailable

# "8 saatten fazla", "more than 8 hours", "50 kilodan az", "under 50 kg"
_MORE = re.compile(r"(\d+(?:[.,]\d+)?)\D{0,12}?(?:fazla|üzeri|üstü|büyük|uzun)|(?:more than|over|above|longer than)\s*(\d+(?:[.,]\d+)?)")
_LESS = re.compile(r"(\d+(?:[.,]\d+)?)\D{0,12}?(?:az|altı|küçük|kısa)|(?:less than|under|below|shorter than)\s*(\d+(?:[.,]\d+)?)")

# column -> (words that select it, Turkish label, English label, unit tr, unit en)
# Checked in this order; the first match wins:
#   charging first: "charges the fastest" also contains "fast" (speed)
#   battery last:  "saat"/"hour" also appear in other questions
COLUMNS = {
    "charge_minutes": (["şarj", "charg"],
                       "şarj süresi", "charging time", "dakika", "minutes"),
    "weight_kg": (["hafif", "ağır", "kilo", "light", "heav", "weigh"],
                  "ağırlık", "weight", "kilogram", "kilograms"),
    "max_speed_ms": (["hızlı", "yavaş", "hız", "fast", "slow", "speed"],
                     "en yüksek hız", "top speed", "metre/saniye", "meters per second"),
    "battery_hours": (["çalış", "batarya", "pil", "dayan", "saat", "battery", "last", "run", "hour"],
                      "batarya ile çalışma süresi", "battery runtime", "saat", "hours"),
}


def build_database(csv_path: str | Path, db_path: str | Path) -> int:
    """(Re)create the robots table from the CSV. Returns the number of robots."""
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute("DROP TABLE IF EXISTS robots")
        db.execute("CREATE TABLE robots (name TEXT PRIMARY KEY, "
                   + ", ".join(f"{c} REAL" for c in COLUMNS) + ")")
        db.executemany(f"INSERT INTO robots VALUES (?{', ?' * len(COLUMNS)})",
                       [[r["name"], *(float(r[c]) if r[c] else None for c in COLUMNS)] for r in rows])
    return len(rows)


class RobotSpecsTool(Tool):
    name = "robot_specs"
    examples = [
        "en hafif robotunuz hangisi", "en hızlı robot hangisi", "en uzun çalışan robot hangisi",
        "hangi robot en kısa sürede şarj olur", "en ağır robot hangisi",
        "robotları hızlarına göre karşılaştır", "hangi robotlar 8 saatten fazla çalışır",
        "which is the lightest robot", "which robot is the fastest", "which robot runs the longest",
        "which robot charges the fastest", "compare the robots by weight",
    ]
    # Comparison words: a question about one robot ("Ada-7 kaç kilo?") must stay on RAG.
    keywords = [" en ", "hangi robot", "hangileri", "karşılaştır", "fazla", "daha",
                "which robot", "which is the", "most", "est ", "compare", "than", "more", "less"]

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run(self, question: str, language: str) -> str:
        column = self._column(question)
        if column is None:  # "En akıllı robot?" - no such column; let RAG try
            raise ToolNotApplicable(question)

        label, unit = self._label(column, language)
        if (flt := self._number_filter(question)) is not None:
            op, limit = flt
            # `column` comes from the COLUMNS whitelist; `limit` is passed as a parameter.
            rows = self._query(f"SELECT name, {column} FROM robots WHERE {column} {op} ? ORDER BY {column}",
                               language, (limit,))
            names = ", ".join(f"{name} ({value:g} {unit})" for name, value in rows)
            word = ("fazla" if op == ">" else "az") if language == "tr" else ("more" if op == ">" else "less")
            if language == "tr":
                return (f"{label.capitalize()} {limit:g} {unit} değerinden {word} olan robotlar: {names}."
                        if rows else f"{label.capitalize()} {limit:g} {unit} değerinden {word} olan robot yok.")
            return (f"Robots with {label} {word} than {limit:g} {unit}: {names}."
                    if rows else f"No robot has {label} {word} than {limit:g} {unit}.")

        # `column` comes from the COLUMNS whitelist, never from user text.
        rows = self._query(f"SELECT name, {column} FROM robots "
                           f"WHERE {column} IS NOT NULL ORDER BY {column}", language)
        values = ", ".join(f"{name} {value:g} {unit}" for name, value in rows)
        (low_name, low), (high_name, high) = rows[0], rows[-1]
        if language == "tr":
            return (f"Robotların {label} değerleri, küçükten büyüğe: {values}.\n"
                    f"En düşük {label}: {low_name} ({low:g} {unit}). En yüksek {label}: {high_name} ({high:g} {unit}).")
        return (f"Robot {label}, from lowest to highest: {values}.\n"
                f"Lowest {label}: {low_name} ({low:g} {unit}). Highest {label}: {high_name} ({high:g} {unit}).")

    def _query(self, sql: str, language: str, params: tuple = ()) -> list[tuple]:
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as db:  # read-only
                return db.execute(sql, params).fetchall()
        except sqlite3.Error as e:
            raise ToolUnavailable("Robot veritabanına şu anda ulaşılamıyor." if language == "tr"
                                  else "The robot database is not available right now.") from e

    @staticmethod
    def _label(column: str, language: str) -> tuple[str, str]:
        _, label_tr, label_en, unit_tr, unit_en = COLUMNS[column]
        return (label_tr, unit_tr) if language == "tr" else (label_en, unit_en)

    @staticmethod
    def _number_filter(question: str) -> tuple[str, float] | None:
        text = question.lower()
        for op, pattern in ((">", _MORE), ("<", _LESS)):
            if m := pattern.search(text):
                return op, float((m.group(1) or m.group(2)).replace(",", "."))
        return None

    @staticmethod
    def _column(question: str) -> str | None:
        text = question.replace("I", "ı").replace("İ", "i").lower()
        for column, (words, *_) in COLUMNS.items():
            if any(w in text for w in words):
                return column
        return None
