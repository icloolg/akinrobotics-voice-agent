"""Admin panel API with a fake retriever (no models, no index needed).

    pytest tests/test_admin.py

Checks the token, file name/type checks and that uploads/deletes stay
inside the knowledge folder.
"""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.admin import create_router, safe_name


class FakeRetriever:
    def chunk_counts(self):
        return {"a.md": 2}


class FakeIndexer:
    def __init__(self):
        self.retriever = FakeRetriever()
        self.requests = 0

    def request(self):
        self.requests += 1


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    monkeypatch.chdir(tmp_path)  # confirmed topics are written to data/topics.json
    (tmp_path / "kb").mkdir()
    (tmp_path / "kb" / "a.md").write_text("# A\n\n## x\ny", encoding="utf-8")
    (tmp_path / "outside.md").write_text("keep me", encoding="utf-8")
    cfg = {"rag": {"knowledge_dir": str(tmp_path / "kb")}}
    indexer = FakeIndexer()
    app = FastAPI()
    app.include_router(create_router(cfg, {"indexer": indexer}))
    c = TestClient(app)
    c.indexer = indexer
    return c, tmp_path


H = {"X-Admin-Token": "secret"}


def test_token_required(client, monkeypatch):
    c, _ = client
    assert c.get("/admin/api/sources").status_code == 401
    assert c.get("/admin/api/sources", headers={"X-Admin-Token": "wrong"}).status_code == 401
    monkeypatch.delenv("ADMIN_TOKEN")
    assert c.get("/admin/api/sources", headers=H).status_code == 503  # panel off without a token


def test_sources_show_chunk_counts(client):
    c, _ = client
    docs = c.get("/admin/api/sources", headers=H).json()["documents"]
    assert [(d["name"], d["chunks"]) for d in docs] == [("a.md", 2)]


def test_upload_stays_in_knowledge_folder(client):
    c, tmp = client
    r = c.post("/admin/api/upload?name=../../yeni belge.MD", headers=H, content="# Yeni".encode())
    assert r.json() == {"saved": "yeni_belge.md", "replaced": False}
    assert (tmp / "kb" / "yeni_belge.md").exists()
    assert c.indexer.requests == 1  # indexing starts by itself


def test_upload_rejects_bad_files(client):
    c, _ = client
    assert c.post("/admin/api/upload?name=x.exe", headers=H, content=b"x").status_code == 400
    assert c.post("/admin/api/upload?name=x.pdf", headers=H, content=b"not a pdf").status_code == 400
    assert c.post("/admin/api/upload?name=x.md", headers=H, content=b"\xff\xfe\x00").status_code == 400


def test_delete_only_indexed_documents(client):
    c, tmp = client
    assert c.delete("/admin/api/sources/..%2Foutside.md", headers=H).status_code == 404
    assert (tmp / "outside.md").exists()
    assert c.delete("/admin/api/sources/a.md", headers=H).status_code == 200
    assert not (tmp / "kb" / "a.md").exists()


def test_safe_name():
    assert safe_name("C:\\Users\\x\\Katalog 2026.PDF") == "Katalog_2026.pdf"
    with pytest.raises(HTTPException):
        safe_name("..md")


def test_confirmed_topics_are_stored(client):
    c, tmp = client
    r = c.put("/admin/api/topics", headers=H, json={"confirmed": ["Mini Ada Kılavuzu", " ", "Mini Ada Kılavuzu"]})
    assert r.json() == {"confirmed": ["Mini Ada Kılavuzu"]}
    assert c.get("/admin/api/topics", headers=H).json()["confirmed"] == ["Mini Ada Kılavuzu"]
