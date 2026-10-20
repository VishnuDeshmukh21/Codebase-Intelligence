# app/auth_cli.py
"""
Command Line Interface (CLI) for Codebase Intelligence MCP Server Authentication.

Enables developers to manage plug-and-play API keys with zero UI.

Commands:
    python -m app.auth_cli create-key --name "Slack Bot" --tenant "engineering"
    python -m app.auth_cli list-keys
    python -m app.auth_cli revoke-key <key_id>
    python -m app.auth_cli verify-key <key>
"""
from __future__ import annotations

import argparse
import os
import sys

# Safe UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from pymongo import MongoClient

# Ensure app directory is discoverable
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

from app.auth_mcp import MCPAuthManager


def get_db():
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        client.admin.command("ping")
        return client["prode_engine"]
    except Exception:
        # Fall back to local key file store
        return None


def handle_create(args):
    db = get_db()
    manager = MCPAuthManager(db)
    raw_key, info = manager.generate_api_key(name=args.name, tenant_id=args.tenant)

    storage_target = "MongoDB Atlas" if db is not None else "Local Key Store"
    print("\n" + "=" * 68)
    print("  [KEY] CODEGRAPH MCP API KEY CREATED")
    print("=" * 68)
    print(f"  Key ID      : {info['_id']}")
    print(f"  Name        : {info['name']}")
    print(f"  Tenant      : {info['tenant_id']}")
    print(f"  Status      : {info['status']}")
    print(f"  Storage     : {storage_target}")
    print(f"  Created At  : {info['created_at']}")
    print("-" * 68)
    print(f"  SECRET KEY  : {raw_key}")
    print("=" * 68)
    print("  [!] IMPORTANT: Copy this secret key now. It will not be shown again.")
    print("  Configure it in your MCP client config under 'env':")
    print(f'    "CODEGRAPH_API_KEY": "{raw_key}"\n')


def handle_list(args):
    db = get_db()
    manager = MCPAuthManager(db)
    keys = manager.list_api_keys(tenant_id=args.tenant)

    if not keys:
        print("\n[i] No API keys found.\n")
        return

    print("\n" + "=" * 80)
    print(f"  {'KEY ID':<18} {'NAME':<20} {'TENANT':<14} {'STATUS':<10} {'CALLS':<8} {'PREFIX'}")
    print("=" * 80)
    for k in keys:
        print(
            f"  {k['_id']:<18} {k.get('name', '')[:19]:<20} {k.get('tenant_id', '')[:13]:<14} "
            f"{k.get('status', ''):<10} {k.get('total_calls', 0):<8} {k.get('key_prefix', '')}"
        )
    print("=" * 80 + "\n")


def handle_revoke(args):
    db = get_db()
    manager = MCPAuthManager(db)
    success = manager.revoke_api_key(args.key_id)
    if success:
        print(f"\n[+] API Key '{args.key_id}' successfully revoked.\n")
    else:
        print(f"\n[-] Key '{args.key_id}' not found or already revoked.\n", file=sys.stderr)


def handle_verify(args):
    db = get_db()
    manager = MCPAuthManager(db)
    is_valid, err, doc = manager.verify_api_key(args.key)
    if is_valid:
        print("\n[+] Key is VALID!")
        print(f"    Name      : {doc.get('name')}")
        print(f"    Tenant    : {doc.get('tenant_id')}")
        print(f"    Status    : {doc.get('status')}")
        print(f"    Calls     : {doc.get('total_calls')}\n")
    else:
        print(f"\n[-] Key is INVALID: {err}\n", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Codebase Intelligence MCP API Key Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-key
    p_create = subparsers.add_parser("create-key", help="Generate a new MCP API key")
    p_create.add_argument("--name", default="MCP Client", help="Human-readable client/agent name")
    p_create.add_argument("--tenant", default="default", help="Tenant or organization identifier")
    p_create.set_defaults(func=handle_create)

    # list-keys
    p_list = subparsers.add_parser("list-keys", help="List registered MCP API keys")
    p_list.add_argument("--tenant", default=None, help="Filter by tenant ID")
    p_list.set_defaults(func=handle_list)

    # revoke-key
    p_revoke = subparsers.add_parser("revoke-key", help="Revoke an existing MCP API key")
    p_revoke.add_argument("key_id", help="Key ID or prefix to revoke")
    p_revoke.set_defaults(func=handle_revoke)

    # verify-key
    p_verify = subparsers.add_parser("verify-key", help="Verify if an API key is valid")
    p_verify.add_argument("key", help="Raw secret key to verify")
    p_verify.set_defaults(func=handle_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

