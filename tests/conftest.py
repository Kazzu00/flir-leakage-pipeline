"""All tests are synthetic and must stay independent of network services."""

import socket

import pytest


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail attempted network connections and tell HF clients to use local files."""
    def reject_connection(*args, **kwargs):
        raise AssertionError("Tests must not access the network or download models")

    monkeypatch.setattr(socket.socket, "connect", reject_connection)
    monkeypatch.setattr(socket, "create_connection", reject_connection)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
