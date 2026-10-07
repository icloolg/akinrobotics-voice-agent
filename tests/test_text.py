"""SentenceSplitter: LLM token stream -> speakable sentences.

    pytest tests/test_text.py
"""
from app.text import SentenceSplitter, clean_for_speech


def speak(tokens: list[str], **kwargs) -> list[str]:
    splitter = SentenceSplitter(**kwargs)
    out = [s for token in tokens for s in splitter.push(token)]
    return out + splitter.flush()


def test_sentence_is_ready_before_the_answer_ends():
    splitter = SentenceSplitter()
    assert splitter.push("Ada-7 65 kilogramdır. Boyu ") == ["Ada-7 65 kilogramdır."]
    assert splitter.flush() == ["Boyu"]


def test_decimal_point_is_not_a_sentence_end():
    assert speak(["En yüksek hızı saniyede 2.5 metredir. ", "Başka?"]) == \
        ["En yüksek hızı saniyede 2.5 metredir.", "Başka?"]


def test_short_pieces_wait_for_the_next_sentence():
    assert speak(["Evet. ", "ARAT merdiven çıkabilir."]) == ["Evet. ARAT merdiven çıkabilir."]


def test_long_sentence_is_cut_at_a_comma():
    text = ("Ada-7 eğitim, turizm, sağlık, perakende ve etkinlik alanlarında kullanılır, "
            "ayrıca on beşten fazla dili konuşabilir ve duyguları ayırt edebilir")
    pieces = speak([w + " " for w in text.split()], max_chars=60)
    assert len(pieces) > 1
    assert all(p.endswith(",") for p in pieces[:-1])
    assert " ".join(pieces).split() == text.split()   # nothing lost or repeated


def test_markdown_and_list_markers_are_not_spoken():
    assert clean_for_speech("- **Ada-7**") == "Ada-7"
    assert speak(["1. Eğitim\n", "2. Sağlık\n"]) == ["Eğitim, Sağlık"]
