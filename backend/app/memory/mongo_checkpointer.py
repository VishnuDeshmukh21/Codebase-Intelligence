# memory/mongo_checkpointer.py
"""
MongoDB-backed LangGraph checkpointer.

LangGraph's checkpointing system persists the full agent state between
invocations, enabling:
  - Resumable sessions (HITL approval doesn't lose context)
  - Conversation history across multiple user messages
  - Debugging: inspect any intermediate state by thread_id

We use langgraph-checkpoint-mongodb if available, falling back to the
MemorySaver for local development without a MongoDB connection.
"""
from __future__ import annotations

import structlog

log = structlog.get_logger()


def build_mongo_checkpointer(mongo_uri: str, client=None):
    """
    Attempts to create a MongoDBSaver checkpointer.
    Falls back to MemorySaver if the package isn't installed or the
    connection fails — so local dev still works without Atlas.

    Parameters
    ----------
    mongo_uri : MongoDB connection string (Atlas or local)
    client    : Optional existing MongoClient instance

    Returns
    -------
    A LangGraph-compatible checkpointer instance.
    """
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient

        mongo_client = client or MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        checkpointer = MongoDBSaver(
            client=mongo_client,
            db_name="prode_engine",
            checkpoint_collection_name="agent_checkpoints",
            writes_collection_name="agent_checkpoint_writes",
        )
        log.info("checkpointer.mongodb_connected", uri=mongo_uri[:40] + "...")
        return checkpointer

    except ImportError:
        log.warning(
            "checkpointer.mongodb_package_missing",
            hint="pip install langgraph-checkpoint-mongodb",
        )
    except Exception as exc:
        log.warning("checkpointer.mongodb_connection_failed", error=str(exc))

    # Graceful fallback — in-memory, no persistence
    from langgraph.checkpoint.memory import MemorySaver

    log.info("checkpointer.using_memory_saver")
    return MemorySaver()

