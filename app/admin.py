"""Admin panel API: knowledge sources and follow-up topics, without a restart.

    ADMIN_TOKEN=<secret> uvicorn app.main:app ...   then open /admin

Adding, replacing or deleting a document starts background re-indexing
(app.knowledge.Indexer); the assistant keeps answering from the previous index
until the new one is swapped in. Topic names suggested from the documents are
confirmed here and take effect for new conversations immediately.

Models and tools are only displayed: they are loaded and warmed up at startup,
so changing them is a config.yaml edit plus a restart.

Every endpoint requires the X-Admin-Token header (compared in constant time);
without ADMIN_TOKEN in the environment the panel is disabled.
"""
import asyncio
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import FileResponse

from app.knowledge import DOCUMENT_TYPES, Indexer, knowledge_dirs, list_documents
from app.knowledge.topics import TopicStore, active_topics

MAX_UPLOAD_MB = 20
ADMIN_PAGE = Path(__file__).parent / "static" / "admin.html"


def create_router(cfg: dict, state: dict) -> APIRouter:
    router = APIRouter()
    target = Path(cfg["rag"]["knowledge_dir"])  # new documents are stored here
    topics = TopicStore()

    def check(token: str | None) -> None:
        expected = os.environ.get("ADMIN_TOKEN")
        if not expected:
            raise HTTPException(503, "Admin paneli kapalı: ADMIN_TOKEN ortam değişkenini ayarlayın.")
        if not token or not secrets.compare_digest(token, expected):
            raise HTTPException(401, "Geçersiz anahtar.")

    def indexer() -> Indexer:
        if "indexer" not in state:
            raise HTTPException(503, "Sunucu henüz yükleniyor.")
        return state["indexer"]

    def document(name: str) -> Path:
        """An indexed document by file name; paths outside the knowledge folders are unreachable."""
        for path in list_documents(cfg):
            if path.name == name:
                return path
        raise HTTPException(404, f"Belge bulunamadı: {name}")

    @router.get("/admin", include_in_schema=False)
    def page():
        return FileResponse(ADMIN_PAGE)

    @router.get("/admin/api/sources")
    async def sources(x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        chunks = await asyncio.to_thread(indexer().retriever.chunk_counts)
        docs = [{
            "name": p.name,
            "folder": p.parent.as_posix(),
            "kb": round(p.stat().st_size / 1024, 1),
            "modified": datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m.%Y %H:%M"),
            "chunks": chunks.get(p.name, 0),  # 0 = not indexed yet
        } for p in list_documents(cfg)]
        return {"documents": docs, "total_chunks": sum(chunks.values()),
                "folders": [d.as_posix() for d in knowledge_dirs(cfg)]}

    @router.post("/admin/api/upload")
    async def upload(request: Request, name: str, x_admin_token: str | None = Header(None)):
        # The file is the raw request body (fetch(url, {body: file})): no multipart parser needed.
        check(x_admin_token)
        path = target / safe_name(name)
        body = await request.body()
        if not body:
            raise HTTPException(400, "Dosya boş.")
        if len(body) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Dosya {MAX_UPLOAD_MB} MB'tan büyük olamaz.")
        if path.suffix == ".pdf" and not body.startswith(b"%PDF"):
            raise HTTPException(400, "Geçerli bir PDF değil.")
        if path.suffix != ".pdf":
            try:
                body.decode("utf-8")
            except UnicodeDecodeError:
                raise HTTPException(400, "Metin dosyası UTF-8 olmalı.")
        target.mkdir(parents=True, exist_ok=True)
        replaced = path.exists()
        path.write_bytes(body)
        indexer().request()
        return {"saved": path.name, "replaced": replaced}

    @router.post("/admin/api/url")
    async def add_url(data: dict, x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        from app.knowledge.web import fetch_page, slug
        url = str(data.get("url", "")).strip()
        try:
            markdown, paragraphs = await asyncio.to_thread(
                fetch_page, url, cfg.get("web_sources", {}).get("min_line_chars", 25))
        except (ValueError, PermissionError) as e:
            raise HTTPException(400, str(e))
        except Exception as e:  # network errors, HTTP 4xx/5xx
            raise HTTPException(502, f"Sayfa indirilemedi: {e}")
        if not paragraphs:
            raise HTTPException(422, "Sayfada ana metin bulunamadı.")
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"web_{slug(url)}.md"
        path.write_text(markdown, encoding="utf-8")
        indexer().request()
        return {"saved": path.name, "paragraphs": paragraphs}

    # Endpoints that start indexing are async: Indexer.request() needs the event loop.
    @router.delete("/admin/api/sources/{name}")
    async def delete(name: str, x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        path = document(name)
        path.unlink()
        indexer().request()
        return {"deleted": path.name}

    @router.post("/admin/api/reindex")
    async def reindex(x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        indexer().request()
        return {"started": True}

    @router.get("/admin/api/status")
    def status(x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        return vars(indexer().status)

    @router.get("/admin/api/topics")
    def get_topics(x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        return {"configured": cfg.get("followup", {}).get("topics", []), "confirmed": topics.load()}

    @router.put("/admin/api/topics")
    def put_topics(data: dict, x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        names = [str(n)[:60] for n in data.get("confirmed", []) if str(n).strip()]
        topics.save(names)
        if "agent" in state:
            state["agent"].set_topics(active_topics(cfg, topics))
        return {"confirmed": topics.load()}

    @router.get("/admin/api/components")
    def components(x_admin_token: str | None = Header(None)):
        check(x_admin_token)
        stt, llm, tts, rag = cfg["stt"], cfg["llm"], cfg["tts"], cfg["rag"]
        verify = cfg.get("verify", {})
        return {
            "STT": f"{stt['provider']} · {stt['model']} ({stt['device']}, {stt['compute_type']})",
            "LLM": f"{llm['provider']} · {llm['model']}",
            "TTS": f"{tts['provider']} · " + ", ".join(f"{k}: {v}" for k, v in tts["voices"].items()),
            "Embedding": rag["embedding_model"],
            "Arama": f"{rag.get('search', 'hybrid')} · top_k {rag['top_k']} · min_score {rag['min_score']}",
            "Doğrulama (NLI)": f"açık · eşik {verify.get('threshold')}" if verify.get("enabled") else "kapalı",
            "Araçlar": ", ".join(cfg.get("tools", {}).get("enabled", [])) or "yok",
        }

    return router


def safe_name(name: str) -> str:
    """A plain file name with an allowed extension (no directories)."""
    name = Path(name.replace("\\", "/")).name
    stem, suffix = Path(name).stem, Path(name).suffix.lower()
    if suffix not in DOCUMENT_TYPES:
        raise HTTPException(400, f"Desteklenen türler: {', '.join(DOCUMENT_TYPES)}")
    stem = re.sub(r"[^\w.-]+", "_", stem).strip("._")
    if not stem:
        raise HTTPException(400, "Geçersiz dosya adı.")
    return stem[:100] + suffix
