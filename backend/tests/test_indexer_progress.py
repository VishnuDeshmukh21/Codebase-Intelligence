from __future__ import annotations

from app.indexers import cpg_indexer


class FakeUpdateOne:
    def __init__(self, filter, update, upsert=False):
        self.filter = filter
        self.update = update
        self.upsert = upsert


class FakeCollection:
    def __init__(self):
        self.documents = {}

    def bulk_write(self, operations, **kwargs):
        modified = 0
        for operation in operations:
            symbol_id = operation.filter["_id"]
            doc = self.documents.get(symbol_id, {"_id": symbol_id})
            if "$or" in operation.filter:
                if "embedding" in doc and doc["embedding"] is not None:
                    continue
            doc.update(operation.update.get("$set", {}))
            for field in operation.update.get("$unset", {}):
                doc.pop(field, None)
            self.documents[symbol_id] = doc
            modified += 1

        return type("BulkResult", (), {"modified_count": modified})()

    def create_index(self, name):
        return name

    def count_documents(self, filter):
        if not filter:
            return len(self.documents)
        return sum(
            1
            for doc in self.documents.values()
            if "embedding" not in doc or doc["embedding"] is None
        )

    def find(self, filter):
        return [
            dict(doc)
            for doc in self.documents.values()
            if "embedding" not in doc or doc["embedding"] is None
            if "file_path" not in filter or doc.get("file_path") == filter["file_path"]
        ]

    def distinct(self, field, filter=None):
        documents = self.documents.values()
        if filter:
            documents = [
                doc for doc in documents
                if "embedding" not in doc or doc["embedding"] is None
            ]
        return list({doc.get(field) for doc in documents})


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
    monkeypatch.setattr(cpg_indexer, "UpdateOne", FakeUpdateOne)
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
    assert updates[-1]["phase"] == "parsed"
    assert updates[-1]["files_completed"] == 2
    assert updates[-1]["percent"] == 20


def test_embedding_stage_batches_missing_symbols_and_resumes(tmp_path, monkeypatch):
    from app import embedding_service

    collection = FakeCollection()
    collection.documents = {
        f"symbol_{index}": {
            "_id": f"symbol_{index}",
            "type": "function",
            "name": f"symbol_{index}",
            "file_path": "module.py",
        }
        for index in range(5)
    }
    monkeypatch.setattr(cpg_indexer, "MongoClient", FakeMongoClient)
    monkeypatch.setattr(cpg_indexer, "UpdateOne", FakeUpdateOne)
    batches = []

    def embed(texts, batch_size):
        batches.append(len(texts))
        return [[float(len(batches))] for _ in texts]

    monkeypatch.setattr(embedding_service, "generate_embeddings", embed)
    stop = {"requested": False}

    def progress(update):
        if update.get("phase") == "embedding":
            stop["requested"] = True

    indexer = cpg_indexer.RepoGraphIndexer(
        repo_path=str(tmp_path),
        repo_id="resume_test",
        progress_callback=progress,
    )
    indexer.symbols_col = collection

    first_run = indexer.generate_missing_embeddings(
        batch_size=2,
        should_stop=lambda: stop["requested"],
    )
    assert first_run["stopped"] is True
    assert first_run["embedding_symbols_completed"] == 2
    assert first_run["embedding_files_completed"] == 0
    assert first_run["embedding_files_total"] == 1
    assert first_run["remaining"] == 3

    stop["requested"] = False
    indexer.progress_callback = None
    second_run = indexer.generate_missing_embeddings(
        batch_size=2,
        should_stop=lambda: stop["requested"],
    )
    assert second_run["stopped"] is False
    assert second_run["embedding_symbols_completed"] == 5
    assert second_run["embedding_files_completed"] == 1
    assert second_run["remaining"] == 0
    assert batches == [2, 2, 1]
