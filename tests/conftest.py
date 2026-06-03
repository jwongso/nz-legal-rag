"""Shared fixtures for tenancy test suite."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def app_client():
    from tenancy.app import app
    with TestClient(app) as client:
        client.headers.update({"X-No-Log": "1"})
        yield client
