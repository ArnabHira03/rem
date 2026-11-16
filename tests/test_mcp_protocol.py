import importlib.metadata
import io
import json

from rem.mcp.protocol import SUPPORTED_VERSIONS, Tool, ToolResult, handle, serve

INSTRUCTIONS = "use rem for memory"


def _echo(args: dict) -> ToolResult:
    return ToolResult(text=f"echo {args.get('x', '')}")


def _boom(args: dict) -> ToolResult:
    raise ValueError("kaboom")


def _soft_fail(args: dict) -> ToolResult:
    return ToolResult(text="nope", is_error=True)


TOOLS = [
    Tool(name="echo", description="Echo x", input_schema={"type": "object", "properties": {"x": {"type": "string"}}}, handler=_echo),
    Tool(name="boom", description="Raises", input_schema={"type": "object"}, handler=_boom),
    Tool(name="soft", description="Errors", input_schema={"type": "object"}, handler=_soft_fail),
]


def run(lines: list[str] | str | bytes) -> list[dict]:
    if isinstance(lines, bytes):
        stdin = io.TextIOWrapper(io.BytesIO(lines), encoding="utf-8", errors="strict")
    else:
        text = lines if isinstance(lines, str) else "".join(line + "\n" for line in lines)
        stdin = io.StringIO(text)
    stdout = io.StringIO()
    assert serve(TOOLS, stdin, stdout, instructions=INSTRUCTIONS) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def req(id_, method, params=None) -> str:
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return json.dumps(msg)


def test_initialize():
    [resp] = run([req(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}})])
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"] == {
        "protocolVersion": "2025-06-18",
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "rem", "version": importlib.metadata.version("rem")},
        "instructions": INSTRUCTIONS,
    }


def test_newer_client_version_gets_ours():
    [resp] = run([req(1, "initialize", {"protocolVersion": "2099-01-01"})])
    assert resp["result"]["protocolVersion"] == "2025-11-25"
    assert SUPPORTED_VERSIONS[0] == "2025-11-25"


def test_ping_with_string_id_echoed():
    [resp] = run([req("abc-1", "ping")])
    assert resp == {"jsonrpc": "2.0", "id": "abc-1", "result": {}}


def test_tools_list_in_registration_order():
    [resp] = run([req(2, "tools/list")])
    assert resp["result"] == {
        "tools": [
            {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            for t in TOOLS
        ]
    }


def test_tools_call_known_tool():
    [resp] = run([req(3, "tools/call", {"name": "echo", "arguments": {"x": "hi"}})])
    assert resp["result"] == {"content": [{"type": "text", "text": "echo hi"}], "isError": False}


def test_tools_call_missing_arguments_is_empty_dict():
    [resp] = run([req(3, "tools/call", {"name": "echo"})])
    assert resp["result"]["content"][0]["text"] == "echo "


def test_tools_call_handler_is_error():
    [resp] = run([req(3, "tools/call", {"name": "soft", "arguments": {}})])
    assert resp["result"] == {"content": [{"type": "text", "text": "nope"}], "isError": True}


def test_tools_call_handler_exception():
    [resp] = run([req(4, "tools/call", {"name": "boom", "arguments": {}})])
    assert resp["result"] == {"content": [{"type": "text", "text": "ValueError: kaboom"}], "isError": True}


def test_tools_call_unknown_tool():
    [resp] = run([req(5, "tools/call", {"name": "nope", "arguments": {}})])
    assert resp == {"jsonrpc": "2.0", "id": 5, "error": {"code": -32602, "message": "unknown tool: nope"}}


def test_tools_call_missing_name():
    out = run([
        req(5, "tools/call", {"arguments": {}}),
        req(6, "tools/call"),
        req(7, "tools/call", {"name": 3}),
    ])
    assert [r["id"] for r in out] == [5, 6, 7]
    assert [r["error"]["code"] for r in out] == [-32602, -32602, -32602]


def test_notification_gets_no_output():
    out = run([
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "method": "ping"}),
        json.dumps({"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "boom"}}),
    ])
    assert out == []


def test_unknown_method():
    [resp] = run([req(7, "resources/list")])
    assert resp["id"] == 7
    assert resp["error"]["code"] == -32601


def test_invalid_json():
    [resp] = run(["{not json"])
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700


def test_invalid_utf8():
    data = b'{"jsonrpc":"2.0","id":1,"method":"ping","x":"\xff\xfe"}\n'
    [resp] = run(data)
    assert resp["id"] is None
    assert resp["error"]["code"] == -32700


def test_non_object_json():
    out = run(["[1, 2]", json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]), "42", '"x"'])
    assert len(out) == 4
    for resp in out:
        assert resp["id"] is None
        assert resp["error"]["code"] == -32600


def test_requests_answered_before_initialize():
    out = run([req(1, "tools/list"), req(2, "ping")])
    assert [r["id"] for r in out] == [1, 2]
    assert all("result" in r for r in out)


def test_blank_lines_skipped():
    out = run("\n   \n" + req(1, "ping") + "\n\n")
    assert out == [{"jsonrpc": "2.0", "id": 1, "result": {}}]


def test_keeps_serving_after_bad_line():
    out = run(["garbage", req(9, "ping")])
    assert len(out) == 2
    assert out[0]["error"]["code"] == -32700
    assert out[1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_keeps_serving_after_bad_utf8_line():
    data = b"\xff\n" + req(9, "ping").encode() + b"\n"
    out = run(data)
    assert out[0]["error"]["code"] == -32700
    assert out[1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_stdout_only_json_lines():
    stdout = io.StringIO()
    stdin = io.StringIO("".join(line + "\n" for line in [
        req(1, "initialize", {"protocolVersion": "2025-11-25"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        req(2, "tools/list"),
        req(3, "tools/call", {"name": "echo", "arguments": {"x": "multi\nline ü"}}),
        req(4, "tools/call", {"name": "boom"}),
        "bad",
        "[]",
        req(5, "nope"),
    ]))
    serve(TOOLS, stdin, stdout, instructions="line1\nline2")
    raw = stdout.getvalue()
    assert raw.endswith("\n")
    lines = raw.splitlines()
    assert len(lines) == 7
    for line in lines:
        msg = json.loads(line)
        assert isinstance(msg, dict)
        assert msg["jsonrpc"] == "2.0"
    assert "ü" in raw  # ensure_ascii=False


def test_handle_returns_none_for_notification():
    assert handle({"jsonrpc": "2.0", "method": "ping"}, {t.name: t for t in TOOLS}, INSTRUCTIONS) is None


def test_handle_null_id_is_a_request():
    resp = handle({"jsonrpc": "2.0", "id": None, "method": "ping"}, {}, INSTRUCTIONS)
    assert resp == {"jsonrpc": "2.0", "id": None, "result": {}}


def test_binary_stdout_is_utf8_regardless_of_text_encoding():
    raw = io.BytesIO()
    stdout = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
    stdin = io.StringIO(req("ü", "ping") + "\n" + '{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}\n')
    assert serve(TOOLS, stdin, stdout, instructions=INSTRUCTIONS) == 0
    lines = raw.getvalue().decode("utf-8").splitlines()
    assert json.loads(lines[0]) == {"jsonrpc": "2.0", "id": "ü", "result": {}}
    assert json.loads(lines[1])["id"] == "\ud800"


def test_deeply_nested_line_is_parse_error_and_serving_continues():
    out = run(["[" * 200000, req(1, "ping")])
    assert len(out) == 2
    assert out[0]["id"] is None
    assert out[0]["error"]["code"] == -32700
    assert out[1] == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_huge_int_line_is_parse_error_and_serving_continues():
    huge = '{"jsonrpc":"2.0","id":' + "9" * 5000 + ',"method":"ping"}'
    out = run([huge, req(2, "ping")])
    assert len(out) == 2
    assert out[0]["id"] is None
    assert out[0]["error"]["code"] == -32700
    assert out[1] == {"jsonrpc": "2.0", "id": 2, "result": {}}
