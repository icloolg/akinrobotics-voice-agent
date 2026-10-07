"""Voice client: microphone -> server, server audio -> speakers.

    python client/voice_client.py                       # local server
    python client/voice_client.py --url ws://HOST:8000/ws

Only needs: numpy, sounddevice, websockets. All AI runs on the server, the
client is just "ears and mouth" (like a sensor node on a robot).

Half-duplex: the mic is muted while the agent is answering, otherwise the
agent would hear itself through the speakers.
"""
import argparse
import asyncio
import json
import queue
import threading

import numpy as np
import sounddevice as sd
import websockets

MIC_RATE = 16000
BLOCK = 512  # 32 ms


class Player:
    """Plays int16 PCM chunks in order on a background thread."""

    def __init__(self, sample_rate: int):
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        self.stream.start()
        threading.Thread(target=self._run, daemon=True).start()

    def play(self, pcm: bytes) -> None:
        self.queue.put(pcm)

    def wait(self) -> None:
        self.queue.join()

    def _run(self) -> None:
        while True:
            pcm = self.queue.get()
            self.stream.write(np.frombuffer(pcm, dtype=np.int16))
            self.queue.task_done()


async def main(url: str) -> None:
    loop = asyncio.get_running_loop()
    mic: asyncio.Queue[bytes] = asyncio.Queue()
    muted = False

    def on_mic(indata, _frames, _time, _status):
        loop.call_soon_threadsafe(mic.put_nowait, bytes(indata))

    async with websockets.connect(url, max_size=None) as ws:
        hello = json.loads(await ws.recv())
        player = Player(hello["sample_rate"])

        async def send_mic():
            while True:
                data = await mic.get()
                if not muted:
                    await ws.send(data)

        async def receive():
            nonlocal muted
            async for msg in ws:
                if isinstance(msg, bytes):
                    player.play(msg)
                    continue
                event = json.loads(msg)
                kind = event["type"]
                if kind == "end_of_speech":
                    muted = True
                    print("  ...")
                elif kind == "transcript":
                    print(f"Sen  ({event['language']}): {event['text']}")
                elif kind == "sentence":
                    print(f"Ajan     : {event['text']}")
                elif kind in ("done", "empty"):
                    if kind == "done":
                        print_metrics(event["metrics"])
                    await asyncio.to_thread(player.wait)   # let the answer finish playing
                    while not mic.empty():                 # drop audio recorded meanwhile
                        mic.get_nowait()
                    muted = False
                    print("\n🎤 Dinliyorum...")

        with sd.InputStream(samplerate=MIC_RATE, channels=1, dtype="int16",
                            blocksize=BLOCK, callback=on_mic):
            print("Bağlandı. 🎤 Dinliyorum... (çıkmak için Ctrl+C)")
            await asyncio.gather(send_mic(), receive())


def print_metrics(m: dict) -> None:
    ms = m["ms"]
    print(f"  [kaynak: {', '.join(m['sources']) or '-'} | STT {ms.get('stt_done', 0):.0f} ms"
          f" | LLM ilk token {ms.get('llm_first_token', 0):.0f} ms"
          f" | ilk ses {ms.get('first_audio', 0):.0f} ms | toplam {ms.get('done', 0):.0f} ms]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    try:
        asyncio.run(main(parser.parse_args().url))
    except KeyboardInterrupt:
        pass
