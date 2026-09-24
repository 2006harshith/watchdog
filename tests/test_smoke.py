import watchdog_agent


def test_version_is_string():
    assert isinstance(watchdog_agent.__version__, str)
