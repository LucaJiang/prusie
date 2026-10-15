"""Evidence and subprocess mocks; no network request is made by these tests."""

import json
import subprocess
import threading
from pathlib import Path
from typing import ClassVar

import pytest

from prusie.ai import CodexCLI, InterpretationError, validate_interpretation
from prusie.report import evidence_bundle


def sample():
    bundle = evidence_bundle({"variants": [{"variant_id": "toy", "pip": 0.75}]})
    identifier = next(f["id"] for f in bundle["facts"] if f["path"].endswith("/pip"))
    response = {
        "claims": [
            {
                "text": "The variant has substantial marginal support.",
                "evidence_ids": [identifier],
                "uncertainty": "This does not establish causality.",
                "suggested_checks": ["Inspect LD compatibility."],
                "numeric_evidence_ids": [identifier],
            }
        ],
        "limitations": ["Toy evidence only."],
    }
    return bundle, response


def test_valid_reply_numbers_rendered_only_from_references():
    bundle, response = sample()
    result = validate_interpretation(json.dumps(response), bundle)
    assert result["claims"][0]["numeric_facts"][0]["value"] == 0.75
    assert result["evidence_sha256"] == bundle["sha256"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda r: r["claims"][0]["evidence_ids"].append("ev_invented"), "Unknown evidence"),
        (lambda r: r["claims"][0].update(text="PIP is 0.99"), "Numbers"),
        (lambda r: r.update(extra="unrequested"), "only claims"),
        (lambda r: r["claims"][0].update(numeric_evidence_ids=["ev_fake"]), "Unknown evidence"),
    ],
)
def test_reject_hallucinated_numbers_references_and_schema(mutation, match):
    bundle, response = sample()
    mutation(response)
    with pytest.raises(InterpretationError, match=match):
        validate_interpretation(response, bundle)


def test_invalid_json_and_tampered_fact():
    bundle, response = sample()
    with pytest.raises(InterpretationError, match=r"Invalid JSON"):
        validate_interpretation("not JSON", bundle)
    bundle["facts"][0]["value"] = "forged"
    with pytest.raises(InterpretationError, match=r"does not match"):
        validate_interpretation(response, bundle)


class MockProcess:
    instances: ClassVar[list] = []
    mode = "valid"
    response = None

    def __init__(self, command, **kwargs):
        self.command, self.kwargs = command, kwargs
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.instances.append(self)

    def communicate(self, input=None, timeout=None):
        if self.mode == "timeout":
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.mode == "failure":
            self.returncode = 1
            return None, None
        output = Path(self.command[self.command.index("--output-last-message") + 1])
        output.write_text(json.dumps(self.response))
        model = self.command[self.command.index("--model") + 1]
        provider = json.loads(
            next(v.split("=", 1)[1] for v in self.command if v.startswith("model_provider="))
        )
        self.kwargs["stdout"].write(
            json.dumps({"type": "turn.completed", "model": model, "provider": provider}) + "\n"
        )
        self.returncode = 0
        return None, None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_no_consent_no_process_or_files(tmp_path):
    bundle, _ = sample()
    called = []
    adapter = CodexCLI(tmp_path / "unused", _popen=lambda *a, **k: called.append(True))
    assert adapter.interpret(bundle)["status"] == "not_authorized"
    assert not called and not (tmp_path / "unused").exists()


def test_mock_official_cli_minimal_environment_and_schema(tmp_path, monkeypatch):
    import prusie.ai as ai

    bundle, response = sample()
    MockProcess.mode, MockProcess.response = "valid", response
    monkeypatch.setattr(ai.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "official-login-opaque"))
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("PYTHONPATH", "/private/project")
    result = CodexCLI(
        tmp_path,
        model="test-model-override",
        provider_config={"name": "OpenAI", "wire_api": "responses", "requires_openai_auth": True},
        _popen=MockProcess,
    ).interpret(bundle, consent=True)
    assert result["status"] == "completed" and result["actual_model"] == "test-model-override"
    child = MockProcess.instances[-1]
    assert child.kwargs["env"]["CODEX_HOME"].endswith("official-login-opaque")
    assert "OPENAI_API_KEY" not in child.kwargs["env"] and "PYTHONPATH" not in child.kwargs["env"]
    assert "--ignore-user-config" in child.command and "read-only" in child.command
    assert (
        "features.shell_tool=false" in child.command and "features.plugins=false" in child.command
    )
    assert 'web_search="disabled"' in child.command and "project_doc_max_bytes=0" in child.command
    assert "features.fast_mode=false" in child.command and 'service_tier="default"' in child.command
    assert 'model_provider="OpenAI"' in child.command
    assert not (tmp_path / "official-login-opaque").exists()
    assert Path(result["workspace"]).parent == tmp_path


def test_mock_timeout_terminates_only_owned_child(tmp_path, monkeypatch):
    import prusie.ai as ai

    bundle, response = sample()
    MockProcess.mode, MockProcess.response = "timeout", response
    monkeypatch.setattr(ai.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "opaque"))
    result = CodexCLI(
        tmp_path, model="test-model", provider="openai", timeout=0.002, _popen=MockProcess
    ).interpret(bundle, consent=True)
    assert result["status"] == "timeout" and result["attempts"] == 1
    assert MockProcess.instances[-1].terminated


def test_cancelled_before_start_and_bounded_attempts(tmp_path):
    event = threading.Event()
    event.set()
    bundle, _ = sample()
    assert CodexCLI(tmp_path).interpret(bundle, consent=True, cancel=event)["status"] == "cancelled"
    with pytest.raises(ValueError, match=r"one or two"):
        CodexCLI(tmp_path, max_attempts=3)


def test_custom_provider_fails_closed_without_safe_config(tmp_path):
    bundle, _ = sample()
    result = CodexCLI(tmp_path, model="test-model").interpret(bundle, consent=True)
    assert result["status"] == "unavailable" and not result["transmitted"]
    assert result["provider_requested"] == "OpenAI"
    with pytest.raises(ValueError, match=r"non-secret"):
        CodexCLI(tmp_path, provider_config={"env_key": "SECRET"})
    with pytest.raises(ValueError, match=r"official OpenAI"):
        CodexCLI(
            tmp_path,
            provider_config={
                "name": "OpenAI",
                "wire_api": "responses",
                "requires_openai_auth": True,
                "base_url": "https://example.com/v1",
            },
        )


def test_finite_retry_failure_and_inflight_cancellation(tmp_path, monkeypatch):
    import prusie.ai as ai

    bundle, response = sample()
    monkeypatch.setattr(ai.shutil, "which", lambda name: "/usr/bin/codex")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "opaque"))
    MockProcess.mode, MockProcess.response = "failure", response
    before = len(MockProcess.instances)
    result = CodexCLI(
        tmp_path, model="test-model", provider="openai", max_attempts=2, _popen=MockProcess
    ).interpret(bundle, consent=True)
    assert result["status"] == "failed" and result["attempts"] == 2
    assert len(MockProcess.instances) == before + 2
    event = threading.Event()

    class CancellingProcess(MockProcess):
        def communicate(self, input=None, timeout=None):
            event.set()
            raise subprocess.TimeoutExpired(self.command, timeout)

    result = CodexCLI(
        tmp_path, model="test-model", provider="openai", _popen=CancellingProcess
    ).interpret(bundle, consent=True, cancel=event)
    assert result["status"] == "cancelled" and MockProcess.instances[-1].terminated


def test_unspecified_model_requires_configuration_without_startup(tmp_path, monkeypatch):
    import prusie.ai as ai

    bundle, _ = sample()

    def unexpected(*args, **kwargs):
        raise AssertionError("No CLI discovery or startup is needed without a model")

    monkeypatch.setattr(ai.shutil, "which", unexpected)
    directory = tmp_path / "unused"
    result = CodexCLI(directory, provider="openai", _popen=unexpected).interpret(
        bundle, consent=True
    )
    assert result["status"] == "unavailable"
    assert result["model_requested"] is None
    assert "Set model explicitly" in result["reason"]
    assert result["attempts"] == 0 and result["transmitted"] is False
    assert not directory.exists()


@pytest.mark.parametrize("model", ["", "   ", 123, False])
def test_invalid_model_identifiers_rejected(tmp_path, model):
    with pytest.raises(ValueError, match=r"model must be"):
        CodexCLI(tmp_path, model=model)
