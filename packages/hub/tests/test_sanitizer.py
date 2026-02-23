"""Sanitizer 硬规则脱敏测试."""

import pytest

from trajectoryhub.sanitizer import sanitize, sanitize_trajectory, SanitizeResult


class TestSanitize:
    """sanitize() 文本级脱敏测试."""

    def test_empty_text(self):
        r = sanitize("")
        assert r.text == ""
        assert r.redacted_count == 0

    def test_no_match(self):
        r = sanitize("Hello, this is a normal text.")
        assert r.text == "Hello, this is a normal text."
        assert r.redacted_count == 0

    def test_sk_token(self):
        r = sanitize("use token sk-abcdefghijklmnopqrstuvwxyz1234567890 here")
        assert "sk-" not in r.text
        assert "[REDACTED_TOKEN]" in r.text
        assert r.redacted_count >= 1

    def test_ghp_token(self):
        r = sanitize("ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn")
        assert "ghp_" not in r.text
        assert "[REDACTED_TOKEN]" in r.text

    def test_phone_number(self):
        r = sanitize("请联系 13812345678 了解详情")
        assert "13812345678" not in r.text
        assert "[REDACTED_PHONE]" in r.text

    def test_email(self):
        r = sanitize("发送到 test@example.com 邮箱")
        assert "test@example.com" not in r.text
        assert "[REDACTED_EMAIL]" in r.text

    def test_id_card(self):
        r = sanitize("身份证号 110101199001011234")
        assert "110101199001011234" not in r.text
        assert "[REDACTED_ID_CARD]" in r.text

    def test_internal_ip(self):
        r = sanitize("服务器在 10.0.1.25 上运行")
        assert "10.0.1.25" not in r.text
        assert "[REDACTED_INTERNAL_IP]" in r.text

    def test_192_168_ip(self):
        r = sanitize("地址 192.168.1.100")
        assert "192.168.1.100" not in r.text
        assert "[REDACTED_INTERNAL_IP]" in r.text

    def test_internal_url(self):
        r = sanitize("访问 https://api.internal.company.com/v2/data")
        assert "internal.company.com" not in r.text
        assert "[REDACTED_INTERNAL_URL]" in r.text

    def test_credential_assignment(self):
        r = sanitize("password=mysecretpassword123")
        assert "mysecretpassword123" not in r.text
        assert "[REDACTED_CREDENTIAL]" in r.text
        # 前缀 password= 应该保留
        assert "password=" in r.text
        assert r.text == "password=[REDACTED_CREDENTIAL]"

    def test_credential_assignment_with_quotes(self):
        r = sanitize("token='my_secret_token_value'")
        assert "my_secret_token_value" not in r.text
        assert "token=" in r.text
        assert "[REDACTED_CREDENTIAL]" in r.text

    def test_credential_assignment_colon(self):
        r = sanitize("secret: mysecretvalue99")
        assert "mysecretvalue99" not in r.text
        assert "secret: " in r.text
        assert "[REDACTED_CREDENTIAL]" in r.text

    def test_long_random_string(self):
        token = "A" * 40
        r = sanitize(f"bearer {token}")
        assert token not in r.text
        assert "[REDACTED_TOKEN]" in r.text

    def test_file_path_not_redacted(self):
        """文件路径不应被 long_random_string 规则误杀."""
        path = "/Users/liukai/knowlyr-gym/packages/hub/src/trajectoryhub"
        r = sanitize(f"reading file {path}")
        assert path in r.text
        assert "[REDACTED_TOKEN]" not in r.text

    def test_multiple_matches(self):
        text = "token sk-abc123456789012345678901 phone 13912345678 email a@b.com"
        r = sanitize(text)
        assert "sk-" not in r.text
        assert "13912345678" not in r.text
        assert "a@b.com" not in r.text
        assert r.redacted_count >= 3

    def test_audit_log_structure(self):
        r = sanitize("call 13800138000")
        assert r.redacted_count >= 1
        d = r.to_dict()
        assert "audit_log" in d
        entry = d["audit_log"][0]
        assert "rule" in entry
        assert "original" in entry
        assert "replacement" in entry
        assert "position" in entry


class TestSanitizeTrajectory:
    """sanitize_trajectory() 轨迹级脱敏测试."""

    def _make_traj(self, thought="", output="", description=""):
        return {
            "task": {"task_id": "t1", "description": description, "domain": "crew"},
            "agent": "crew/test",
            "model": "test-model",
            "steps": [
                {
                    "step_id": 1,
                    "thought": thought,
                    "tool_call": {
                        "name": "Bash",
                        "parameters": {"command": "echo hello"},
                    },
                    "tool_result": {
                        "output": output,
                        "exit_code": 0,
                    },
                }
            ],
            "metadata": {},
        }

    def test_no_sensitive_data(self):
        traj = self._make_traj(thought="分析代码", output="done")
        clean = sanitize_trajectory(traj)
        assert clean["steps"][0]["thought"] == "分析代码"
        assert clean["steps"][0]["tool_result"]["output"] == "done"
        # 没有敏感数据时不应有审计日志
        assert "sanitize_audit" not in clean.get("metadata", {})

    def test_thought_sanitized(self):
        traj = self._make_traj(thought="token is sk-abcdefghij0123456789012345")
        clean = sanitize_trajectory(traj)
        assert "sk-" not in clean["steps"][0]["thought"]
        assert "[REDACTED_TOKEN]" in clean["steps"][0]["thought"]

    def test_output_sanitized(self):
        traj = self._make_traj(output="email: test@secret.com")
        clean = sanitize_trajectory(traj)
        assert "test@secret.com" not in clean["steps"][0]["tool_result"]["output"]
        assert "[REDACTED_EMAIL]" in clean["steps"][0]["tool_result"]["output"]

    def test_description_sanitized(self):
        traj = self._make_traj(description="联系 13800001111")
        clean = sanitize_trajectory(traj)
        assert "13800001111" not in clean["task"]["description"]
        assert "[REDACTED_PHONE]" in clean["task"]["description"]

    def test_original_not_modified(self):
        traj = self._make_traj(thought="secret sk-abcdefghij0123456789012345")
        original_thought = traj["steps"][0]["thought"]
        _ = sanitize_trajectory(traj)
        # 原始对象不应被修改
        assert traj["steps"][0]["thought"] == original_thought

    def test_audit_log_in_metadata(self):
        traj = self._make_traj(
            thought="token sk-abcdefghij0123456789012345",
            output="ip is 10.0.0.1",
        )
        clean = sanitize_trajectory(traj)
        audit = clean["metadata"]["sanitize_audit"]
        assert audit["total_redacted"] >= 2
        assert len(audit["entries"]) >= 2

    def test_flat_step_format(self):
        """兼容扁平格式的 step (tool/params/output 直接在 step 上)."""
        traj = {
            "task": {"task_id": "t1", "description": "test"},
            "steps": [
                {
                    "step_id": 1,
                    "tool": "Bash",
                    "params": {"command": "curl -H 'token=secret12345678'"},
                    "output": "response from 192.168.1.1",
                }
            ],
            "metadata": {},
        }
        clean = sanitize_trajectory(traj)
        assert "192.168.1.1" not in clean["steps"][0]["output"]
