from harness_core import credentials


def test_local_source_key_is_available(monkeypatch):
    monkeypatch.delenv("PYTRIO_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_local_values", lambda: {"PYTRIO_API_KEY": "local_dummy_key"})
    assert credentials.get_secret("PYTRIO_API_KEY") == "local_dummy_key"


def test_environment_takes_precedence(monkeypatch):
    monkeypatch.setenv("PYTRIO_API_KEY", "environment_dummy_key")
    monkeypatch.setattr(credentials, "_local_values", lambda: {"PYTRIO_API_KEY": "local_dummy_key"})
    assert credentials.get_secret("PYTRIO_API_KEY") == "environment_dummy_key"
