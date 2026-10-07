import logging

import requests

from app.tools.base import Tool, ToolUnavailable

log = logging.getLogger(__name__)

STATUS_TR = {"working": "görevde", "charging": "şarj oluyor", "idle": "beklemede", "error": "arızalı"}


class RobotStatusTool(Tool):
    """Live robot fleet status from an HTTP API (battery, status, location).

    In this project the API is a mock served by the voice server itself
    (/mock/robots, see app/mock_robot_api.py); in a real deployment the URL
    would point to the fleet management system (e.g. Roboliza).
    """

    name = "robot_status"
    # Live state ("şu an", "now") — not fixed specs like "kaç saat çalışır",
    # which are knowledge questions and must stay on the RAG path.
    examples = [
        "robotların şu anki durumu ne", "robot şu an ne yapıyor", "robot şu an nerede",
        "şu an şarjı yüzde kaç", "şu anda hangi robot şarj oluyor", "şu an hangi robotlar görevde",
        "arızalı robot var mı", "robotların anlık batarya seviyesi",
        "what is the robots' status right now", "what is the robot doing now", "where is the robot now",
        "what is its battery percentage right now", "which robots are charging at the moment",
        "is any robot broken",
    ]

    # Live-state words. Measured on 10 status + 10 spec questions written with
    # different wording than the examples: without this rule "Robotlarınız neler
    # yapabilir?" (a spec question) scored 0.931 and went to the tool.
    # With it: 8/10 status questions -> tool, 0/10 spec questions -> tool.
    keywords = ["şu an", "şuan", "şimdi", "anlık", "durum", "arıza",
                "now", "currently", "at the moment", "status", "broken"]

    def __init__(self, url: str, timeout_s: float = 2.0):
        self.url = url
        self.timeout_s = timeout_s

    def run(self, question: str, language: str) -> str:
        try:
            response = requests.get(self.url, timeout=self.timeout_s)
            response.raise_for_status()
            robots = response.json()["robots"]
        except Exception as e:  # network error, HTTP error, bad JSON, ...
            log.warning("Robot status API failed: %r", e)
            raise ToolUnavailable("Robot durum servisine şu anda ulaşılamıyor." if language == "tr"
                                  else "The robot status service is not reachable right now.") from e

        if language == "tr":
            lines = [f"{r['name']}: batarya yüzde {r['battery']}, {STATUS_TR.get(r['status'], r['status'])}, "
                     f"konum {r['location']}." for r in robots]
            return "Anlık robot filosu durumu:\n" + "\n".join(lines)
        lines = [f"{r['name']}: battery {r['battery']} percent, {r['status']}, location {r['location']}."
                 for r in robots]
        return "Live robot fleet status:\n" + "\n".join(lines)
