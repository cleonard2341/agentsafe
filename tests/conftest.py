import pytest
from agentsafe.storage.database import Database
from agentsafe.storage.repository import EventRepository


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def repo(db):
    return EventRepository(db)
