"""Summarise the measured turns (logs/turns.jsonl) and print the hardware.

    python -m scripts.benchmark                 # every logged turn
    python -m scripts.benchmark --last 30       # only the 30 most recent turns
    python -m scripts.benchmark --markdown      # table ready to paste into README.md

The server appends one line per finished voice turn. All times are
milliseconds; t=0 is the moment the VAD decided the user stopped speaking.

  stt_done         speech recognised
  llm_first_token  first piece of the answer text
  first_audio      first sentence synthesised (ready to send to the client)
  response         first_audio + the VAD's silence wait (vad.min_silence_ms):
                   user stopped speaking -> answer audio. The number the user feels.
  done             whole answer synthesised
RTF (real-time factor) = processing time / audio duration; below 1 is faster than real time.
"""
import argparse
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path

from app.metrics import TURNS_FILE

ROUTES = {"rag": "Bilgi (RAG)", "tool": "Araç", "chitchat": "Sohbet", "no_answer": "Cevap yok (LLM'siz)"}


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile; with few samples p95 is close to the maximum."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]


def load(path: Path, last: int | None) -> list[dict]:
    turns = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return turns[-last:] if last else turns


def route_of(turn: dict) -> str:
    # Older log lines have no "route"; they were RAG answers or fixed "no information" answers.
    return turn.get("route") or ("rag" if turn.get("grounded") else "no_answer")


def response_ms(turn: dict, default_endpoint: float) -> float:
    return turn.get("response_ms") or turn["ms"]["first_audio"] + turn.get("endpoint_ms", default_endpoint)


def hardware() -> list[str]:
    lines = [f"İşletim sistemi: {platform.system()} {platform.release()}",
             f"İşlemci: {cpu_name()} ({os.cpu_count()} mantıksal çekirdek)"]
    if ram := ram_gb():
        lines.append(f"Bellek: {ram:.0f} GB")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        lines.append(f"GPU: {gpu or 'yok'}")
    except (OSError, subprocess.SubprocessError):
        lines.append("GPU: yok (nvidia-smi bulunamadı)")
    return lines


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "bilinmiyor"


def ram_gb() -> float | None:
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                           [(name, ctypes.c_ulonglong) for name in
                            ("total", "avail", "total_page", "avail_page", "total_virtual",
                             "avail_virtual", "avail_extended")]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total / 2 ** 30
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
    except (OSError, ValueError, AttributeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=TURNS_FILE)
    parser.add_argument("--last", type=int, help="only the N most recent turns")
    parser.add_argument("--endpoint-ms", type=float, default=500,
                        help="VAD silence wait for old log lines that did not record it")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"{args.file} yok. Önce sunucuyla birkaç sesli tur yapın.")
    turns = load(args.file, args.last)
    if not turns:
        raise SystemExit("Kayıtlı tur yok.")

    print("Donanım")
    for line in hardware():
        print(f"  {line}")

    header = ["Yol", "Tur", "STT", "LLM ilk token", "İlk ses", "Yanıt gecikmesi p50", "p95", "Toplam"]
    rows = []
    groups = [(label, [t for t in turns if route_of(t) == route]) for route, label in ROUTES.items()]
    for label, group in groups + [("Tümü", turns)]:
        if not group:
            continue
        med = lambda key: statistics.median(t["ms"][key] for t in group if key in t["ms"])
        resp = [response_ms(t, args.endpoint_ms) for t in group]
        rows.append([label, str(len(group)), f"{med('stt_done'):.0f}", f"{med('llm_first_token'):.0f}",
                     f"{med('first_audio'):.0f}", f"{statistics.median(resp):.0f}",
                     f"{percentile(resp, 95):.0f}", f"{med('done'):.0f}"])

    print(f"\nGecikme (ms, medyan; {len(turns)} tur)")
    if args.markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
        for row in rows:
            print("| " + " | ".join(row) + " |")
    else:
        widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
        for row in [header] + rows:
            print("  " + "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                                   for i, (c, w) in enumerate(zip(row, widths))))

    stt = [t["stt_rtf"] for t in turns if t.get("stt_rtf")]
    tts = [t["tts_rtf"] for t in turns if t.get("tts_rtf")]
    print("\nRTF (işlem süresi / ses süresi)")
    print(f"  STT: ortalama {statistics.mean(stt):.3f}, en kötü {max(stt):.3f}"
          f"  (soru sesi ort. {statistics.mean(t['user_audio_s'] for t in turns):.1f} sn)")
    print(f"  TTS: ortalama {statistics.mean(tts):.3f}, en kötü {max(tts):.3f}")


if __name__ == "__main__":
    main()
