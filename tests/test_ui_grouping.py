"""Tests for the UI trace grouping (pure functions, no Streamlit calls)."""

from app.ui import _tool_label, group_trace


def test_group_command_family_into_one_step():
    events = [
        {"type": "command", "text": "python analysis.py"},
        {"type": "output", "text": "row 1\nrow 2"},
        {"type": "error", "text": "warning"},
        {"type": "status", "ok": False, "code": 3},
    ]
    steps = group_trace(events)
    assert len(steps) == 1
    step = steps[0]
    assert step["kind"] == "command"
    assert step["command"] == "python analysis.py"
    assert step["output"] == "row 1\nrow 2"
    assert step["error"] == "warning"
    assert step["status"] == {"ok": False, "code": 3}


def test_group_tool_pair_into_one_step():
    events = [
        {"type": "tool_call", "name": "inspect_dataset", "args": "{'path': 'x'}", "id": "t1"},
        {"type": "tool_result", "text": "shape info", "id": "t1"},
    ]
    steps = group_trace(events)
    assert len(steps) == 1
    assert steps[0] == {
        "kind": "tool",
        "id": "t1",
        "name": "inspect_dataset",
        "args": "{'path': 'x'}",
        "result": "shape info",
        "awaiting_result": False,
    }


def test_group_pairs_results_by_tool_call_id():
    """Interleaved multi-call messages: results must attach to the right
    step via tool_call_id, not positional adjacency."""
    events = [
        {"type": "tool_call", "name": "inspect_dataset", "args": "{a}", "id": "A"},
        {"type": "tool_call", "name": "ls", "args": "{b}", "id": "B"},
        {"type": "tool_result", "text": "inspect result", "id": "A"},
        {"type": "tool_result", "text": "ls result", "id": "B"},
    ]
    steps = group_trace(events)
    assert [s["kind"] for s in steps] == ["tool", "tool"]
    assert steps[0]["name"] == "inspect_dataset"
    assert steps[0]["result"] == "inspect result"
    assert steps[1]["name"] == "ls"
    assert steps[1]["result"] == "ls result"


def test_group_command_and_tool_interleaved_by_id():
    events = [
        {"type": "command", "text": "ls", "id": "c1"},
        {"type": "tool_call", "name": "write_file", "args": "{p}", "id": "t1"},
        {"type": "output", "text": "file1", "id": "c1"},
        {"type": "status", "ok": True, "code": 0, "id": "c1"},
        {"type": "tool_result", "text": "saved", "id": "t1"},
    ]
    steps = group_trace(events)
    assert [s["kind"] for s in steps] == ["command", "tool"]
    assert steps[0]["output"] == "file1"
    assert steps[0]["status"] == {"ok": True, "code": 0}
    assert steps[1]["result"] == "saved"


def test_group_preserves_order_and_notes():
    events = [
        {"type": "commentary", "text": "plan"},
        {"type": "command", "text": "ls"},
        {"type": "status", "ok": True, "code": 0},
        {"type": "commentary", "text": "done"},
        {"type": "final", "text": "answer"},
    ]
    steps = group_trace(events)
    assert [s["kind"] for s in steps] == ["note", "command", "note"]
    assert steps[0]["text"] == "plan"
    assert steps[1]["status"] == {"ok": True, "code": 0}
    assert steps[2]["text"] == "done"


def test_orphan_output_becomes_note():
    events = [{"type": "output", "text": "stray"}]
    steps = group_trace(events)
    assert steps == [{"kind": "note", "text": "stray"}]


def test_tool_labels():
    assert _tool_label("inspect_dataset", "") == "Inspected dataset"
    assert _tool_label("ls", "") == "Listed files"
    assert _tool_label("write_file", "{'file_path': 'artifacts/a.py'}") == "Saved artifacts/a.py"
    assert _tool_label("edit_file", "{'file_path': 'artifacts/a.py'}") == "Updated artifacts/a.py"
    assert _tool_label("read_file", "{'file_path': 'data/x.csv'}") == "Read data/x.csv"
    assert _tool_label("custom_tool", "{}") == "custom_tool"
