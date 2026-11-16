"""Hand-rolled stdio MCP server: JSON-RPC 2.0, one JSON object per line.

Stdlib only (the MCP SDK is deliberately not used). Nothing but JSON-RPC
messages is ever written to stdout.
"""

from __future__ import annotations

import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

SUPPORTED_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


@dataclass
class ToolResult:
    text: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], ToolResult]


def _result(id_: object, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: object, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _server_version() -> str:
    try:
        return importlib.metadata.version("rem")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def _initialize(params: dict, instructions: str) -> dict:
    requested = params.get("protocolVersion")
    version = requested if requested in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
    return {
        "protocolVersion": version,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "rem", "version": _server_version()},
        "instructions": instructions,
    }


def _call_tool(id_: object, params: dict, tools: dict[str, Tool]) -> dict:
    name = params.get("name")
    if not isinstance(name, str):
        return _error(id_, INVALID_PARAMS, "params.name must be a string")
    tool = tools.get(name)
    if tool is None:
        return _error(id_, INVALID_PARAMS, f"unknown tool: {name}")
    arguments = params.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return _error(id_, INVALID_PARAMS, "params.arguments must be an object")
    try:
        r = tool.handler(arguments)
    except Exception as exc:  # a tool failure is a tool result, not a protocol error
        r = ToolResult(text=f"{type(exc).__name__}: {exc}", is_error=True)
    return _result(id_, {"content": [{"type": "text", "text": r.text}], "isError": r.is_error})


def handle(message: object, tools: dict[str, Tool], instructions: str) -> dict | None:
    """Answer one parsed message; ``None`` for notifications (no ``"id"``)."""
    if not isinstance(message, dict):
        return _error(None, INVALID_REQUEST, "request must be a JSON object")
    if "id" not in message:
        return None
    id_ = message["id"]
    method = message.get("method")
    if not isinstance(method, str):
        return _error(id_, INVALID_REQUEST, "method must be a string")
    params = message.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        return _error(id_, INVALID_PARAMS, "params must be an object")

    if method == "initialize":
        return _result(id_, _initialize(params, instructions))
    if method == "ping":
        return _result(id_, {})
    if method == "tools/list":
        return _result(id_, {
            "tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
                for t in tools.values()
            ]
        })
    if method == "tools/call":
        return _call_tool(id_, params, tools)
    return _error(id_, METHOD_NOT_FOUND, f"method not found: {method}")


def _lines(stdin: TextIO):
    """Yield raw lines as bytes when the stream has a buffer, else as text."""
    buffer = getattr(stdin, "buffer", None)
    return buffer if buffer is not None else stdin


def serve(tools: list[Tool], stdin: TextIO, stdout: TextIO, *, instructions: str) -> int:
    by_name = {t.name: t for t in tools}
    for raw in _lines(stdin):
        if isinstance(raw, bytes):
            try:
                line = raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                response = _error(None, PARSE_ERROR, "invalid UTF-8")
                _write(stdout, response)
                continue
        else:
            line = raw
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            response = _error(None, PARSE_ERROR, "invalid JSON")
        else:
            response = handle(message, by_name, instructions)
        if response is not None:
            _write(stdout, response)
    return 0


def _write(stdout: TextIO, message: dict) -> None:
    """One line of UTF-8 JSON, flushed. Bytes go to ``stdout.buffer`` when present,
    so the locale's encoding never matters."""
    text = json.dumps(message, ensure_ascii=False)
    try:
        data = (text + "\n").encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates echoed back (e.g. in an id)
        text = json.dumps(message)
        data = (text + "\n").encode("ascii")
    buffer = getattr(stdout, "buffer", None)
    if buffer is not None:
        stdout.flush()
        buffer.write(data)
        buffer.flush()
    else:
        stdout.write(data.decode("utf-8"))
        stdout.flush()
