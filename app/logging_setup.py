import logging


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Third-party libraries are noisy at INFO.
    for name in ("httpx", "chromadb", "sentence_transformers", "faster_whisper"):
        logging.getLogger(name).setLevel(logging.WARNING)
