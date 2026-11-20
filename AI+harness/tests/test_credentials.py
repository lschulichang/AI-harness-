from harness_demo import credentials


def test_local_source_key_is_available(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(credentials, "_local_values", lambda: {"DEEPSEEK_API_KEY": "local_dummy_key"})
    assert credentials.get_secret("DEEPSEEK_API_KEY") == "local_dummy_key"


def test_environment_takes_precedence(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment_dummy_key")
    monkeypatch.setattr(credentials, "_local_values", lambda: {"DEEPSEEK_API_KEY": "local_dummy_key"})
    assert credentials.get_secret("DEEPSEEK_API_KEY") == "environment_dummy_key"
