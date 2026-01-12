"""
Minimal MCP client that connects to an MCP server and lists its tools.

Supports two common transports:
1) STDIO transport (spawn a local server process)
2) SSE/HTTP transport (connect to a remote server URL)

Usage (STDIO):
  python mcp_list_tools.py --stdio -- cmd arg1 arg2 ...

Example:
  python mcp_list_tools.py --stdio -- python -m my_mcp_server

Usage (SSE/HTTP):
  python mcp_list_tools.py --sse http://127.0.0.1:8000/sse

Notes:
- Requires the official MCP Python SDK (package name often: `mcp`).
- The exact import paths can differ slightly depending on SDK version.
"""

import argparse
import asyncio
import json
import sys
from typing import Any, Dict


async def list_tools_stdio(cmd: list[str]) -> None:
    # MCP Python SDK (typical)
    from mcp.client.stdio import stdio_client
    from mcp.client.session import ClientSession
    print("STDIO MCP TOOLS")
    
    async with stdio_client(cmd) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()

            # tools is typically a List[Tool] (pydantic-like). Convert to JSON-friendly.
            json_tools = json.dumps([t.model_dump() if hasattr(t, "model_dump") else t for t in tools])
            print(json_tools)
            with open("./temp.jsonl", "w") as fp:
                print("in here")
                json.dump([t.model_dump() if hasattr(t, "model_dump") else t for t in tools], fp)

async def list_tools_sse(url: str) -> None:
    # MCP Python SDK (typical)
    from fastmcp import Client
    print(f"HTTP MCP TOOLS @ `{url}`")
    
    async with Client(url) as client:
        tools = await client.list_tools()
        print(f"Available tools: {tools}")
        
        with open("./temp.jsonl", "w") as fp:
            print("in here")
            json.dump([t.model_dump() if hasattr(t, "model_dump") else t for t in tools], fp)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List tools from an MCP server (fast minimal client).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--stdio", action="store_true", help="Connect via stdio by spawning a server command.")
    g.add_argument("--sse", type=str, help="Connect via SSE/HTTP to the given server URL (e.g., http://localhost:8000/sse).")
    p.add_argument("--", dest="cmd_sep", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="Server command after --stdio, e.g. -- python -m my_server")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    if args.stdio:
        # Expect command after `--`
        cmd = [c for c in args.cmd if c != "--"]
        if not cmd:
            raise SystemExit("STDIO mode requires a server command, e.g. --stdio -- python -m my_mcp_server")
        await list_tools_stdio(cmd)
        return

    if args.sse:
        await list_tools_sse(args.sse)
        return


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ModuleNotFoundError as e:
        print(
            "Missing dependency. Install the MCP SDK your server uses (commonly: `pip install mcp`).\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        raise
