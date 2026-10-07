"""Tools that need no model: the specs database and the clock.

    pytest tests/test_tools.py
"""
import pytest

from app.tools.base import ToolNotApplicable, ToolUnavailable
from app.tools.datetime_tool import DateTimeTool
from app.tools.robot_specs import RobotSpecsTool, build_database


@pytest.fixture
def specs(tmp_path):
    db = tmp_path / "robots.db"
    assert build_database("knowledge/robot_specs.csv", db) == 8
    return RobotSpecsTool(str(db))


def test_superlative_question_gets_sorted_values(specs):
    context = specs.run("En hafif robotunuz hangisi?", "tr")
    assert "En düşük ağırlık: ARAT (27 kilogram)" in context
    assert "En yüksek ağırlık: AMR (230 kilogram)" in context


def test_charging_is_checked_before_speed(specs):
    # "charges the fastest" contains "fast" but is about charging time.
    assert "charging time" in specs.run("Which robot charges the fastest?", "en")


def test_number_filter_runs_in_sql_not_in_the_llm(specs):
    context = specs.run("Hangi robotlar 8 saatten fazla çalışır?", "tr")
    assert "AMR" in context
    assert "Ada-7" not in context     # exactly 8 hours is not "more than 8"


def test_number_filter_with_no_match(specs):
    assert "No robot" in specs.run("Which robots weigh more than 500 kilograms?", "en")


def test_unknown_attribute_is_left_to_the_documents(specs):
    # "smartest" is not a column: the agent falls back to RAG instead of guessing from the table.
    with pytest.raises(ToolNotApplicable):
        specs.run("En akıllı robot hangisi?", "tr")


def test_missing_database_is_reported_not_guessed(tmp_path):
    with pytest.raises(ToolUnavailable):
        RobotSpecsTool(str(tmp_path / "missing.db")).run("En hızlı robot hangisi?", "tr")


def test_datetime_answers_in_the_question_language():
    tool = DateTimeTool(utc_offset_hours=3)
    assert tool.run("Saat kaç?", "tr").startswith("Şu anki saat")
    assert tool.run("What time is it?", "en").startswith("The current time is")
