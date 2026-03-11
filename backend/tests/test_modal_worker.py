"""Unit tests for ModalWorker — init, register, emit, opencode lifecycle, verify config."""

import subprocess
from unittest.mock import MagicMock, patch, call

import httpx
import pytest

from worker.modal_worker import ModalWorker


@pytest.fixture
def worker():
    return ModalWorker("test-worker-1", "http://localhost:8000", function_call_id="fc-abc123")


# -- Init ----------------------------------------------------------------------


class TestInit:
    def test_defaults(self, worker):
        assert worker.worker_id == "test-worker-1"
        assert worker.service_url == "http://localhost:8000"
        assert worker.function_call_id == "fc-abc123"
        assert worker.opencode_url is None
        assert worker._proc is None

    def test_no_function_call_id(self):
        w = ModalWorker("w1", "http://localhost:8000")
        assert w.function_call_id is None


# -- emit_event ----------------------------------------------------------------


class TestEmitEvent:
    @patch("worker.modal_worker.httpx.post")
    def test_emit_event_posts_to_api(self, mock_post, worker):
        worker.emit_event("worker.started", "Worker ready", {"foo": "bar"})

        mock_post.assert_called_once_with(
            "http://localhost:8000/events",
            json={
                "event_type": "worker.started",
                "summary": "Worker ready",
                "details": {"foo": "bar"},
                "worker_id": "test-worker-1",
            },
            timeout=5.0,
        )

    @patch("worker.modal_worker.httpx.post")
    def test_emit_event_defaults_empty_details(self, mock_post, worker):
        worker.emit_event("worker.heartbeat", "alive")

        body = mock_post.call_args[1]["json"]
        assert body["details"] == {}

    @patch("worker.modal_worker.httpx.post", side_effect=httpx.ConnectError("refused"))
    def test_emit_event_swallows_errors(self, mock_post, worker):
        # Should not raise
        worker.emit_event("worker.started", "Worker ready")


# -- register ------------------------------------------------------------------


class TestRegister:
    @patch("worker.modal_worker.httpx.post")
    def test_register_posts_with_function_call_id(self, mock_post, worker):
        mock_post.return_value = MagicMock(status_code=200)
        worker.opencode_url = "https://tunnel.modal.run"

        worker.register()

        mock_post.assert_called_once_with(
            "http://localhost:8000/workers/register",
            json={
                "worker_id": "test-worker-1",
                "opencode_url": "https://tunnel.modal.run",
                "function_call_id": "fc-abc123",
            },
            timeout=10.0,
        )

    @patch("worker.modal_worker.httpx.post")
    def test_register_raises_on_http_error(self, mock_post, worker):
        mock_post.return_value = MagicMock()
        mock_post.return_value.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=MagicMock()
        )
        worker.opencode_url = "https://tunnel.modal.run"

        with pytest.raises(httpx.HTTPStatusError):
            worker.register()


# -- opencode lifecycle --------------------------------------------------------


class TestOpencodeLifecycle:
    @patch("worker.modal_worker.subprocess.Popen")
    def test_start_opencode(self, mock_popen, worker):
        mock_proc = MagicMock()
        mock_popen.return_value = mock_proc

        worker.start_opencode("/workspace")

        assert worker._proc is mock_proc
        mock_popen.assert_called_once_with(
            ["opencode", "serve", "--hostname", "0.0.0.0", "--port", "4096"],
            stdout=-1,  # subprocess.PIPE
            stderr=-2,  # subprocess.STDOUT
            text=True,
            cwd="/workspace",
        )

    def test_stop_opencode_when_proc_is_none(self, worker):
        worker._proc = None
        worker.stop_opencode()  # should not raise

    def test_stop_opencode_terminates(self, worker):
        mock_proc = MagicMock()
        worker._proc = mock_proc

        worker.stop_opencode()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)

    def test_stop_opencode_kills_on_timeout(self, worker):
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("opencode", 5)
        worker._proc = mock_proc

        worker.stop_opencode()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


# -- wait_for_opencode ---------------------------------------------------------


class TestWaitForOpencode:
    @patch("worker.modal_worker.time.sleep")
    @patch("worker.modal_worker.httpx.get")
    def test_returns_on_success(self, mock_get, mock_sleep, worker):
        mock_get.return_value = MagicMock(status_code=200)

        worker.wait_for_opencode(timeout_s=5.0)
        mock_get.assert_called()

    @patch("worker.modal_worker.time.sleep")
    @patch("worker.modal_worker.time.time")
    @patch("worker.modal_worker.httpx.get", side_effect=httpx.ConnectError("refused"))
    def test_raises_on_timeout(self, mock_get, mock_time, mock_sleep, worker):
        # Simulate time passing beyond deadline
        mock_time.side_effect = [0.0, 0.0, 1.0, 2.0, 3.0]

        with pytest.raises(RuntimeError, match="opencode not ready"):
            worker.wait_for_opencode(timeout_s=1.0)


# -- verify_opencode_config ----------------------------------------------------


class TestVerifyConfig:
    @patch("worker.modal_worker.httpx.get")
    def test_valid_config(self, mock_get, worker):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "provider": {
                    "opencode-go": {
                        "options": {"apiKey": "sk-real-key-123"}
                    }
                }
            }),
        )

        worker.verify_opencode_config()  # should not raise

    @patch("worker.modal_worker.httpx.get")
    def test_missing_provider(self, mock_get, worker):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"provider": {"other": {}}}),
        )

        with pytest.raises(RuntimeError, match="missing 'opencode-go' provider"):
            worker.verify_opencode_config()

    @patch("worker.modal_worker.httpx.get")
    def test_unresolved_api_key(self, mock_get, worker):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "provider": {
                    "opencode-go": {
                        "options": {"apiKey": "{env:OPENCODE_GO_API_KEY}"}
                    }
                }
            }),
        )

        with pytest.raises(RuntimeError, match="apiKey not resolved"):
            worker.verify_opencode_config()

    @patch("worker.modal_worker.httpx.get")
    def test_empty_api_key(self, mock_get, worker):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "provider": {
                    "opencode-go": {
                        "options": {"apiKey": ""}
                    }
                }
            }),
        )

        with pytest.raises(RuntimeError, match="apiKey not resolved"):
            worker.verify_opencode_config()
