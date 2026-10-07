from datetime import datetime, timedelta, timezone

from app.tools.base import Tool

DAYS = {
    "tr": ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"],
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
}
MONTHS = {
    "tr": ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz",
           "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"],
    "en": ["January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December"],
}


class DateTimeTool(Tool):
    """Current date and time. An LLM cannot know this; asked directly it guesses."""

    name = "datetime"
    examples = [
        "saat kaç", "şu an saat kaç", "bugün günlerden ne", "bugünün tarihi ne",
        "bugün ayın kaçı", "hangi yıldayız",
        "what time is it", "what is the date today", "what day is it today", "what year is it",
    ]

    # "kaç saat çalışır" (a duration) must not reach the clock: measured, "Peki kaç
    # saat çalışır?" went to this tool. Word order matters: "saat kaç", not "kaç saat".
    keywords = ["saat kaç", "saati kaç", "tarih", "gün", "ayın kaçı", "yıl", "yılda",
                "time is it", "the time", "date", "what day", "year"]

    def __init__(self, utc_offset_hours: int = 3):
        # Turkey is UTC+3 all year (no daylight saving since 2016), so a fixed
        # offset is enough and avoids needing a timezone database on Windows.
        self.tz = timezone(timedelta(hours=utc_offset_hours))

    def run(self, question: str, language: str) -> str:
        now = datetime.now(self.tz)
        lang = language if language in DAYS else "en"
        day, month = DAYS[lang][now.weekday()], MONTHS[lang][now.month - 1]
        if lang == "tr":
            return f"Şu anki saat {now:%H:%M}. Bugün {now.day} {month} {now.year}, {day}."
        return f"The current time is {now:%H:%M}. Today is {day}, {month} {now.day}, {now.year}."
