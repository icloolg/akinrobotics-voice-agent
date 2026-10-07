"""Mock robot fleet API — stands in for a real fleet management system.

GET /mock/robots -> {"robots": [{name, battery, status, location}, ...]}

Battery levels change with the clock so answers visibly come from live
data, not from the model's memory.
"""
import time

from fastapi import APIRouter

router = APIRouter()

FLEET = [
    # name, location, status, battery at minute 0, % change per minute
    ("Ada-7 (1)", "Lobi", "working", 90, -0.5),
    ("Ada-7 (2)", "Şarj istasyonu", "charging", 20, +1.0),
    ("Mini Ada", "Toplantı salonu", "idle", 65, -0.1),
    ("Servis Robotu V3", "Kafeterya", "working", 55, -0.4),
    ("ARAT", "Test alanı", "error", 40, 0.0),
]


@router.get("/mock/robots")
def robots():
    minute = (time.time() / 60) % 120  # repeats every 2 hours
    out = []
    for name, location, status, start, rate in FLEET:
        battery = int(min(100, max(5, start + rate * minute)))
        out.append({"name": name, "battery": battery, "status": status, "location": location})
    return {"robots": out}
