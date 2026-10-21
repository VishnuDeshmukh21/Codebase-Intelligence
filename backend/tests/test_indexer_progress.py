from __future__ import annotations

from app.indexers import cpg_indexer


class FakeCollection:
    def bulk_write(self, operations):
        return None

    def create_index(self, name):
        return name


class FakeDatabase:
    def __getitem__(self, name):
        return FakeCollection()


class FakeMongoClient:
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, name):
        return FakeDatabase()

    def close(self):
        pass


def test_indexer_reports_python_file_progress(tmp_path, monkeypatch):
    (tmp_path / "one.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("def second():\n    return 2\n", encoding="utf-8")
    monkeypatch.setattr(cpg_indexer, "MongoClient", FakeMongoClient)
    monkeypatch.setattr(cpg_indexer, "generate_embeddings", None)
    monkeypatch.setattr(cpg_indexer, "build_symbol_embedding_text", None)
    updates = []

    indexer = cpg_indexer.RepoGraphIndexer(
        repo_path=str(tmp_path),
        repo_id="progress_test",
        progress_callback=updates.append,
    )
    indexer.index_repository()

    assert updates[0]["phase"] == "parsing"
    assert updates[0]["total_files"] == 2
    assert any(update["current_file"] for update in updates)
    assert updates[-1]["phase"] == "completed"
    assert updates[-1]["files_completed"] == 2
    assert updates[-1]["percent"] == 100
