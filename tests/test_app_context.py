"""Tests for AppContext dataclass."""

from app.app_context import AppContext


def _noop(*args, **kwargs):
    return None


def _make_ctx(**overrides):
    defaults = dict(
        get_client=lambda: None,
        get_write_locked=lambda: False,
        get_app_config=lambda: None,
        get_calibration=lambda: None,
        get_lamps=lambda: [],
        get_current_tube_type=lambda: "6L6",
        get_current_lamp_id=lambda: "L1",
        set_poller_active=_noop,
    )
    defaults.update(overrides)
    return AppContext(**defaults)


class TestAppContext:
    def test_create_minimal(self):
        ctx = _make_ctx()
        assert ctx.get_current_tube_type() == "6L6"
        assert ctx.get_current_lamp_id() == "L1"
        assert ctx.get_lamps() == []
        assert ctx.get_write_locked() is False

    def test_optional_fields_default_none(self):
        ctx = _make_ctx()
        assert ctx.get_preheat_enabled is None
        assert ctx.get_preheat_done is None
        assert ctx.request_start_preheat is None
        assert ctx.request_stop_all is None
        assert ctx.request_stop_keep_heater is None

    def test_optional_fields_can_be_set(self):
        ctx = _make_ctx(
            get_preheat_enabled=lambda: True,
            get_preheat_done=lambda: False,
            request_start_preheat=_noop,
            request_stop_all=_noop,
            request_stop_keep_heater=_noop,
        )
        assert ctx.get_preheat_enabled() is True
        assert ctx.get_preheat_done() is False
        assert ctx.request_start_preheat() is None

    def test_set_poller_active_callable(self):
        calls = []
        ctx = _make_ctx(set_poller_active=lambda active: calls.append(active))
        ctx.set_poller_active(True)
        ctx.set_poller_active(False)
        assert calls == [True, False]

    def test_client_getter(self):
        sentinel = object()
        ctx = _make_ctx(get_client=lambda: sentinel)
        assert ctx.get_client() is sentinel
