"""Answer verification rules that need no model.

    pytest tests/test_verify.py
"""
from app.agent.verify import Verifier

CONTEXT = ["Ada-7'nin boyu 166 santimetre, ağırlığı 65 kilogramdır. En yüksek hızı saniyede 0,6 metredir.",
           "AKINSOFT'un 4.957 saha personeli vardır ve 1995'te kuruldu."]


def unknown(sentence):
    return Verifier._unknown_numbers(sentence, CONTEXT)


def test_numbers_from_context_pass():
    assert unknown("Ada-7'nin boyu 166 santimetredir.") == []
    assert unknown("AKINSOFT 1995'te kuruldu ve 4.957 saha personeli var.") == []


def test_invented_or_converted_numbers_are_caught():
    # the NLI model alone accepted this sentence (0.97)
    assert unknown("Ada-7'in boyu 166 santimetre veya 1,66 metre olup, bu 5,43 metreye eşittir.") == ["1.66", "5.43"]
    assert unknown("Ada-7 70 kilogramdır.") == ["70"]


def test_decimal_comma_and_point_are_the_same_number():
    assert unknown("Ada-7 saniyede 0.6 metre gider.") == []


def test_digits_inside_names_are_not_numbers():
    assert unknown("Ada-7 ve Servis Robotu V3 sosyal robotlardır.") == []
