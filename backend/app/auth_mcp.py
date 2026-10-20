# app/auth_mcp.py
"""
Plug-and-play Authentication Manager for Codebase Intelligence MCP Server.

Provides API key management and validation without requiring a separate web UI.
Keys are hashed with SHA-256 and stored in MongoDB (`codegraph_api_keys`)
with automatic fallback to a local JSON key store (`storage/keys/api_keys.json`)
if MongoDB is offline.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pymongo.database import Database

log = structlog.get_logger()

KEY_PREFIX = "cg_live_"
STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage" / "keys"
LOCAL_KEYS_FILE = STORAGE_DIR / "api_keys.json"


def hash_key(raw_key: str) -> str:
    """Hashes an API key using SHA-256."""
    return hashlib.sha256(raw_key.strip().encode("utf-8")).hexdigest()


class MCPAuthManager:
    """
    Manages API keys for the Model Context Protocol (MCP) server.
    Ensures that external agents (Slack bot, IDE extensions, custom agents)
    can plug into the MCP server securely.
    """

    def __init__(self, db: Optional[Database] = None):
        self.db = db

    def _get_collection(self):
        if self.db is None:
            return None
        try:
            col = self.db["codegraph_api_keys"]
            return col
        except Exception as exc:
            log.warning("auth_mcp.db_unreachable", error=str(exc))
            return None

    def _read_local_keys(self) -> List[Dict[str, Any]]:
        if not LOCAL_KEYS_FILE.exists():
            return []
        try:
            return json.loads(LOCAL_KEYS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _write_local_keys(self, keys: List[Dict[str, Any]]) -> None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        LOCAL_KEYS_FILE.write_text(json.dumps(keys, indent=2), encoding="utf-8")

    def generate_api_key(
        self,
        name: str = "Default Client",
        tenant_id: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generates a new secure API key, stores its SHA-256 hash in MongoDB
        (or local fallback store), and returns the raw secret key.
        """
        raw_key = f"{KEY_PREFIX}{secrets.token_hex(20)}"
        key_id = f"key_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        doc = {
            "_id": key_id,
            "name": name.strip(),
            "tenant_id": tenant_id.strip(),
            "key_prefix": raw_key[:14] + "...",
            "key_hash": hash_key(raw_key),
            "status": "active",
            "created_at": now,
            "last_used_at": None,
            "total_calls": 0,
            "metadata": metadata or {},
        }

        col = self._get_collection()
        stored_in_mongo = False
        if col is not None:
            try:
                col.insert_one(doc)
                stored_in_mongo = True
                log.info("auth_mcp.key_created_mongo", key_id=key_id, name=name)
            except Exception as exc:
                log.warning("auth_mcp.mongo_insert_failed", error=str(exc))

        # Always also store in local keys store as resilient backup
        local_keys = self._read_local_keys()
        local_keys.append(doc)
        self._write_local_keys(local_keys)

        clean_doc = {k: v for k, v in doc.items() if k != "key_hash"}
        clean_doc["storage"] = "mongodb" if stored_in_mongo else "local_store"
        return raw_key, clean_doc

    def verify_api_key(self, raw_key: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Validates an API key against environment master keys, MongoDB, or local key store.

        Returns:
            (is_valid: bool, error_message: Optional[str], key_data: Optional[Dict])
        """
        # 1. Check if auth is explicitly disabled via environment variable
        auth_required_env = os.environ.get("CODEGRAPH_AUTH_REQUIRED", "").lower()
        if auth_required_env in ("false", "0", "no", "off"):
            return True, None, {"tenant_id": "anonymous", "name": "Auth Disabled", "status": "active"}

        candidate_key = (raw_key or os.environ.get("CODEGRAPH_API_KEY", "")).strip()

        # 2. Check environment master key / local dev bypass
        master_key = os.environ.get("CODEGRAPH_MASTER_KEY", "").strip()
        if master_key and candidate_key == master_key:
            return True, None, {"tenant_id": "master", "name": "Master Environment Key", "status": "active"}

        # 3. If no candidate key provided:
        if not candidate_key:
            col = self._get_collection()
            local_keys = self._read_local_keys()
            total_keys = (col.count_documents({}) if col is not None else 0) + len(local_keys)
            # If no keys exist anywhere and auth_required is not explicitly enforced: bootstrap mode
            if total_keys == 0 and not master_key and auth_required_env != "true":
                return True, None, {"tenant_id": "bootstrap", "name": "Bootstrap Mode", "status": "active"}
            return (
                False,
                "Authentication failed: Missing CODEGRAPH_API_KEY. "
                "Set CODEGRAPH_API_KEY in your environment/MCP config, or create a key using 'python -m app.auth_cli create-key'.",
                None,
            )

        hashed = hash_key(candidate_key)

        # 4. Try MongoDB verification
        col = self._get_collection()
        doc = None
        if col is not None:
            try:
                doc = col.find_one({"key_hash": hashed})
                if doc:
                    try:
                        col.update_one(
                            {"_id": doc["_id"]},
                            {
                                "$set": {"last_used_at": datetime.now(timezone.utc).isoformat()},
                                "$inc": {"total_calls": 1},
                            },
                        )
                    except Exception:
                        pass
            except Exception as exc:
                log.warning("auth_mcp.mongo_query_failed", error=str(exc))

        # 5. Try Local key store fallback if not found in MongoDB
        if not doc:
            local_keys = self._read_local_keys()
            for k in local_keys:
                if k.get("key_hash") == hashed:
                    doc = k
                    k["last_used_at"] = datetime.now(timezone.utc).isoformat()
                    k["total_calls"] = k.get("total_calls", 0) + 1
                    self._write_local_keys(local_keys)
                    break

        if not doc:
            return False, "Authentication failed: Invalid API key.", None

        if doc.get("status") != "active":
            return False, f"Authentication failed: API key has status '{doc.get('status')}'.", None

        clean_doc = {k: v for k, v in doc.items() if k != "key_hash"}
        return True, None, clean_doc

    def list_api_keys(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lists all registered API keys without exposing the secret hash."""
        seen_ids = set()
        keys = []

        # From Mongo
        col = self._get_collection()
        if col is not None:
            try:
                query = {"tenant_id": tenant_id} if tenant_id else {}
                for d in col.find(query, {"key_hash": 0}):
                    seen_ids.add(d["_id"])
                    keys.append(d)
            except Exception:
                pass

        # From local file
        for k in self._read_local_keys():
            if k["_id"] not in seen_ids:
                if tenant_id and k.get("tenant_id") != tenant_id:
                    continue
                clean = {x: y for x, y in k.items() if x != "key_hash"}
                keys.append(clean)

        return keys

    def revoke_api_key(self, key_id_or_prefix: str) -> bool:
        """Revokes an API key by ID or prefix."""
        revoked = False

        # In Mongo
        col = self._get_collection()
        if col is not None:
            try:
                res = col.update_one(
                    {"$or": [{"_id": key_id_or_prefix}, {"key_prefix": {"$regex": f"^{key_id_or_prefix}"}}]},
                    {"$set": {"status": "revoked", "revoked_at": datetime.now(timezone.utc).isoformat()}},
                )
                if res.modified_count > 0:
                    revoked = True
            except Exception:
                pass

        # In local file
        local_keys = self._read_local_keys()
        for k in local_keys:
            if k["_id"] == key_id_or_prefix or k.get("key_prefix", "").startswith(key_id_or_prefix):
                k["status"] = "revoked"
                k["revoked_at"] = datetime.now(timezone.utc).isoformat()
                revoked = True
        self._write_local_keys(local_keys)

        return revoked

