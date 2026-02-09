"""Rule-based reward functions for agent trajectory evaluation.

Each function takes trajectory data and returns scores in [0.0, 1.0].
These functions handle the "rule" evaluator layer of the 3-layer architecture.
"""

from __future__ import annotations

from typing import Any


def _normalize_step(step: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    """从步骤字典中提取 tool/params/output，兼容两种格式.

    格式 1（简单）: {tool, params, output}
    格式 2（recorder）: {tool_call: {name, parameters}, tool_result: {output}}

    Returns:
        (tool_name, params_dict, output_str)
    """
    # 格式 1
    tool = step.get("tool", "")
    params = step.get("params", {})
    output = step.get("output", "")

    # 格式 2: recorder 转换输出
    if not tool and "tool_call" in step:
        tc = step["tool_call"]
        tool = tc.get("name", "")
        params = tc.get("parameters", {})
    if not output and "tool_result" in step:
        tr = step["tool_result"]
        output = tr.get("output", "")

    return tool, params, output


def check_redundancy(steps: list[dict[str, Any]]) -> list[float]:
    """Detect redundant operations in a trajectory.

    Checks for:
    - Repeated file reads of the same file without intervening changes
    - Duplicate tool calls with identical parameters
    - Edit-undo patterns (edit followed by reverting the same content)

    Args:
        steps: List of step dicts, each with at least:
            - tool: str (tool name) 或 tool_call: {name, parameters}
            - params: dict (tool parameters)
            - output: str (tool output, optional) 或 tool_result: {output}

    Returns:
        List of scores per step, 1.0 = non-redundant, 0.0 = fully redundant
    """
    if not steps:
        return []

    scores = []
    seen_calls: list[tuple[str, str]] = []  # (tool, normalized_params)

    for i, step in enumerate(steps):
        tool, params, _output = _normalize_step(step)
        score = 1.0

        # Create a normalized key for this call
        param_key = _normalize_params(params)
        call_key = (tool, param_key)

        # For read-type tools, check if the file was edited since last read
        is_read_tool = tool in ("read_file", "Read", "cat", "grep", "Grep", "Glob")
        edited_since_last_read = False

        if is_read_tool and i > 0:
            file_path = params.get("file_path", params.get("path", ""))
            if file_path:
                last_edit_idx = _find_last_edit_index(steps[:i], file_path)
                last_read_idx = _find_last_read_index(steps[:i], file_path)
                if last_edit_idx is not None and (
                    last_read_idx is None or last_edit_idx > last_read_idx
                ):
                    # File was edited since last read — re-reading is justified
                    edited_since_last_read = True

        # Check for exact duplicate calls
        if call_key in seen_calls and not edited_since_last_read:
            if is_read_tool:
                score = 0.0
            else:
                # Other repeated calls get partial penalty
                score = 0.3

        # Check for read-after-read of the same file without edits in between
        if is_read_tool and i > 0 and not edited_since_last_read:
            file_path = params.get("file_path", params.get("path", ""))
            if file_path:
                last_read_idx = _find_last_read_index(steps[:i], file_path)
                if last_read_idx is not None:
                    # Read the same file again without editing it in between
                    score = 0.0

        scores.append(score)
        seen_calls.append(call_key)

    return scores


def check_efficiency(steps: list[dict[str, Any]], reference_steps: int) -> float:
    """Compute efficiency score based on step count ratio.

    A trajectory that solves the task in fewer steps than reference is more efficient.

    Args:
        steps: List of steps in the trajectory
        reference_steps: Expected/reference number of steps for this task

    Returns:
        Score in [0.0, 1.0], where 1.0 means equal or fewer steps than reference
    """
    if reference_steps <= 0:
        return 1.0

    actual_steps = len(steps)
    if actual_steps == 0:
        return 0.0

    ratio = reference_steps / actual_steps
    # Clamp to [0, 1]: if fewer steps than reference, score = 1.0
    # If more steps, score decreases linearly
    return min(1.0, max(0.0, ratio))


def check_outcome(outcome: dict[str, Any]) -> float:
    """Score based on task outcome (test pass/fail).

    Args:
        outcome: Dict with outcome information:
            - success: bool (whether the task was completed)
            - tests_passed: int (number of tests passed, optional)
            - tests_total: int (total number of tests, optional)
            - partial_credit: float (manual partial credit, optional)

    Returns:
        Score in [0.0, 1.0]
    """
    if not outcome:
        return 0.0

    # If there's a manual partial credit, use it directly
    if "partial_credit" in outcome:
        return max(0.0, min(1.0, float(outcome["partial_credit"])))

    # If test results are available, compute pass rate
    tests_passed = outcome.get("tests_passed", 0)
    tests_total = outcome.get("tests_total", 0)

    if tests_total > 0:
        return tests_passed / tests_total

    # Fall back to boolean success
    if outcome.get("success", False):
        return 1.0

    return 0.0


def check_regression(steps: list[dict[str, Any]]) -> list[float]:
    """Detect edit-then-revert patterns (regressions) in a trajectory.

    Looks for patterns where:
    - A file is edited, then the edit is reverted (same old_string/new_string swapped)
    - Content is written then overwritten back to original

    Args:
        steps: List of step dicts with tool/params/output

    Returns:
        List of scores per step, 1.0 = no regression, 0.0 = full revert
    """
    if not steps:
        return []

    scores = [1.0] * len(steps)
    edits_history: list[dict[str, Any]] = []  # Track past edits

    for i, step in enumerate(steps):
        tool, params, _output = _normalize_step(step)

        if tool in ("Edit", "edit_file", "sed", "write_file", "Write"):
            file_path = params.get("file_path", params.get("path", ""))
            old_string = params.get("old_string", "")
            new_string = params.get("new_string", "")

            if old_string and new_string:
                # Check if this edit reverts a previous one
                for prev_edit in edits_history:
                    if (
                        prev_edit["file_path"] == file_path
                        and prev_edit["old_string"] == new_string
                        and prev_edit["new_string"] == old_string
                    ):
                        # This is a revert of a previous edit
                        scores[i] = 0.0
                        break

                edits_history.append(
                    {
                        "file_path": file_path,
                        "old_string": old_string,
                        "new_string": new_string,
                        "step_index": i,
                    }
                )

    return scores


def check_info_utilization(steps: list[dict[str, Any]]) -> list[float]:
    """Check if each step utilizes information from previous steps.

    Heuristic: If a step's parameters reference content from previous step outputs,
    it's utilizing prior information well.

    Args:
        steps: List of step dicts with tool/params/output

    Returns:
        List of scores per step, 1.0 = good utilization, lower = wasteful
    """
    if not steps:
        return []

    scores = []
    collected_info: set[str] = set()  # Key pieces of info from outputs

    for i, step in enumerate(steps):
        if i == 0:
            # First step has no prior info to use
            scores.append(1.0)
        else:
            tool, params, _step_output = _normalize_step(step)
            score = 1.0

            # If this is a search/read tool, it's gathering info — always OK
            if tool in ("read_file", "Read", "Grep", "Glob", "grep", "find", "ls"):
                score = 1.0
            # If this is an edit/write tool, check if params reference prior info
            elif tool in ("Edit", "edit_file", "Write", "write_file", "Bash"):
                # Check if the step's params contain any info from prior outputs
                params_str = str(params).lower()
                has_reference = False
                for info_piece in collected_info:
                    if info_piece.lower() in params_str:
                        has_reference = True
                        break
                # If editing without referencing any prior findings, slight penalty
                if collected_info and not has_reference:
                    score = 0.7

            scores.append(score)

        # Collect key info from this step's output
        _tool, _params, output = _normalize_step(step)
        if output:
            # Extract file paths, function names, variable names as "info pieces"
            for token in _extract_info_tokens(output):
                collected_info.add(token)

    return scores


# --- Helper functions ---


def _normalize_params(params: dict[str, Any]) -> str:
    """Create a normalized string key from parameters for comparison."""
    # Sort keys and convert to string for consistent comparison
    sorted_items = sorted(params.items())
    return str(sorted_items)


def _find_last_read_index(
    steps: list[dict[str, Any]], file_path: str
) -> int | None:
    """Find the index of the last read of a specific file."""
    for i in range(len(steps) - 1, -1, -1):
        tool, params, _output = _normalize_step(steps[i])
        if tool in ("read_file", "Read"):
            step_path = params.get("file_path", params.get("path", ""))
            if step_path == file_path:
                return i
    return None


def _find_last_edit_index(
    steps: list[dict[str, Any]], file_path: str
) -> int | None:
    """Find the index of the last edit of a specific file."""
    for i in range(len(steps) - 1, -1, -1):
        tool, params, _output = _normalize_step(steps[i])
        if tool in ("Edit", "edit_file", "Write", "write_file"):
            step_path = params.get("file_path", params.get("path", ""))
            if step_path == file_path:
                return i
    return None


def _extract_info_tokens(output: str) -> list[str]:
    """Extract key information tokens from step output.

    Extracts file paths, identifiers, and other useful tokens.
    """
    tokens = []
    # Extract file-path-like tokens
    for word in output.split():
        # File paths
        if "/" in word and len(word) > 3:
            cleaned = word.strip(".,;:()[]{}\"'`")
            if cleaned:
                tokens.append(cleaned)
        # Function/variable names (camelCase or snake_case)
        if "_" in word and len(word) > 3 and word.isidentifier():
            tokens.append(word)

    return tokens
