"""Compare embedding models for retrieval, independent of our thresholds.

    python -m scripts.compare_embeddings
    python -m scripts.compare_embeddings intfloat/multilingual-e5-small BAAI/bge-m3

All thresholds in config.yaml were measured on e5-small's score scale, so
running the full eval with another model would be unfair to it. Instead:
  1. Retrieval accuracy: is the right section 1st / in the top 2 (what the LLM gets)?
  2. Separation: how far the best score of answerable questions stays above
     the best score of off-topic questions (what the "I don't know" threshold
     relies on). Positive gap = one threshold can split them perfectly.
  3. Speed and size on this machine's CPU.
Each model gets its own temporary index; the real one in data/ is not touched.
"""
import sys
import tempfile
import time

from app.agent.retriever import Retriever
from app.config import load_config
from app.knowledge.documents import load_passages

# question -> part of the right section's title ("a|b" = either is right)
LABELED = [
    ("Ada-7 kaç kilo?", "Ada-7 Sosyal"),
    ("How much does Ada-7 weigh?", "Ada-7 Sosyal"),
    ("Mini Ada'nın bataryası kaç saat dayanıyor?", "Mini Ada Sosyal"),
    ("How long does the Mini Ada battery last?", "Mini Ada Sosyal"),
    ("ARAT merdiven çıkabilir mi?", "ARAT Dört"),
    ("Can ARAT climb stairs?", "ARAT Dört"),
    ("AKINCI-5'in boyu kaç santimetre?", "AKINCI-5"),
    ("How fast can AKINCI-5 move?", "AKINCI-5"),
    ("AMR v1000 en fazla kaç kilogram yük taşır?", "AMR Depo"),
    ("UV-C robotu 60 dakikada kaç metrekare alanı temizler?", "UV-C"),
    ("Servis Robotu V3'te kaç tepsi var?", "Servis Robotu V3"),
    ("Fabrika Servis Robotu'nun rafları kaç kilo taşır?", "Fabrika Servis"),
    ("Hangi robotlarınız var?", "Ürün Gamı"),
    ("What robots do you have?", "Ürün Gamı"),
    ("AROS nedir?", "AROS Nedir"),
    ("What is AROS?", "AROS Nedir"),
    ("Roboliza ne işe yarar?", "AROS Bileşenleri"),
    ("AKINSOFT'un kurucusu kimdir?", "AKINSOFT Nedir|Şirket Hakkında"),
    ("AKINSOFT nedir?", "AKINSOFT Nedir"),
    ("AKINSOFT kaç ülkede hizmet veriyor?", "Yaygınlığı"),
    ("What products does AKINSOFT have?", "Yazılım Ürünleri|Bulut"),
    ("İstanbul Robot Müzesi nedir?", "Bağlı Kuruluşlar"),
    ("Fabrikanın kapalı alanı kaç metrekare?", "Şirket Bilgileri - Fabrika"),
    ("Hangi robotları kiralayabilirim?", "Robot Kiralama"),
    ("AKINROBOTICS'in misyonu nedir?", "Misyon"),
    ("Robot kolun taşıma kapasitesi ne kadar?", "robotkol-katalog - Sayfa 3"),
    ("What is the payload capacity of the robot arm?", "robotkol-katalog - Sayfa 3"),
]
OFF_TOPIC = [
    "Bugün hava nasıl?", "Who won the World Cup?", "Türkiye'nin başkenti neresi?", "Bana bir fıkra anlat",
    "What is 2+2?", "Yemek tarifi önerir misin?", "Futbol maçı kaçta?", "How do I cook pasta?",
]
# e5 needs "query: "/"passage: "; other models get no prefix.
PREFIXES = {"e5": ("query: ", "passage: ")}


def evaluate(model_name: str, cfg: dict) -> dict:
    q_prefix, p_prefix = next((v for k, v in PREFIXES.items() if k in model_name.lower()), ("", ""))
    # ignore_cleanup_errors: on Windows Chroma still holds its files open at exit.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        start = time.perf_counter()
        r = Retriever(model_name, tmp, q_prefix, p_prefix)
        load_s = time.perf_counter() - start
        start = time.perf_counter()
        passages, _ = load_passages(cfg)
        texts = [p.text for p in passages]
        chunks = r.replace(texts, [p.source for p in passages], [p.url for p in passages], r.embed_passages(texts))
        index_s = time.perf_counter() - start

        top1 = top2 = 0
        answerable, misses = [], []
        query_times = []
        for question, want in LABELED:
            t = time.perf_counter()
            vector = r.encode_query(question)
            query_times.append(time.perf_counter() - t)
            hits = r.search(question, 2, vector=vector).chunks
            titles = [h.text.splitlines()[0] for h in hits]
            ok = [any(w in title for w in want.split("|")) for title in titles]
            top1 += ok[0]
            top2 += any(ok)
            answerable.append(hits[0].score)
            if not ok[0]:
                misses.append(f"{question} -> {titles[0][-40:]}")
        off = [r.search(q, 1).chunks[0].score for q in OFF_TOPIC]
        params = sum(p.numel() for p in r.model.parameters()) / 1e6
        del r

    return {"model": model_name, "params_M": params, "chunks": chunks, "top1": top1, "top2": top2,
            "n": len(LABELED), "min_answerable": min(answerable), "max_off_topic": max(off),
            "overlap": sum(a <= max(off) for a in answerable), "query_ms": 1000 * sum(query_times) / len(query_times),
            "index_s": index_s, "load_s": load_s, "misses": misses}


if __name__ == "__main__":
    cfg = load_config()
    models = sys.argv[1:] or [cfg["rag"]["embedding_model"], "BAAI/bge-m3"]
    results = [evaluate(m, cfg) for m in models]

    print(f"\n{'model':32s} {'param':>6s} {'1. sıra':>8s} {'ilk 2':>6s} {'en düşük':>9s} {'konu dışı':>10s} "
          f"{'boşluk':>7s} {'çakışan':>8s} {'sorgu':>7s} {'indeks':>7s}")
    for x in results:
        print(f"{x['model']:32s} {x['params_M']:5.0f}M {x['top1']:>4d}/{x['n']:<3d} {x['top2']:>3d}/{x['n']:<2d}"
              f" {x['min_answerable']:9.3f} {x['max_off_topic']:10.3f} {x['min_answerable'] - x['max_off_topic']:+7.3f}"
              f" {x['overlap']:>8d} {x['query_ms']:5.0f}ms {x['index_s']:6.1f}s")
    for x in results:
        if x["misses"]:
            print(f"\n{x['model']} — 1. sırada yanlış bölüm:")
            for m in x["misses"]:
                print("   ", m)
