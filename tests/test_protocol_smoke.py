"""Smoke tests for protocol response parsing."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm19.protocol import (
    LM19Serial,
    ParamValue,
    decode_ih,
    decode_ug1,
    decode_uh,
    encode_ih,
    encode_ug1,
    encode_uh,
)

pytestmark = [pytest.mark.smoke_protocol]


@pytest.mark.smoke
def test_protocol_frame_parse_smoke(monkeypatch):
    client = LM19Serial("COM_FAKE")

    monkeypatch.setattr(client, "_read_until_semicolon", lambda timeout=1.0: "\r\nUa=123;")
    pv = client._read_param_response()
    assert pv.name == "Ua"
    assert pv.value == 123

    monkeypatch.setattr(client, "_read_until_semicolon", lambda timeout=1.0: "bad_response;")
    with pytest.raises(ValueError):
        client._read_param_response()


@pytest.mark.smoke
def test_reopen_survives_serialexception_on_close(monkeypatch):
    """reopen() must not raise NameError when ser.close() throws
    SerialException — that is exactly the USB-reconnect path it exists for."""
    import serial

    client = LM19Serial("COM_FAKE")

    class _DeadSer:
        def close(self):
            raise serial.SerialException("device disconnected")

    client.ser = _DeadSer()
    opened = {"v": False}

    def _fake_open():
        opened["v"] = True
        client.ser = object()

    monkeypatch.setattr(client, "open", _fake_open)

    client.reopen()  # previously raised NameError: SerialException not imported

    assert opened["v"] is True


@pytest.mark.smoke
def test_protocol_encode_decode_roundtrip_smoke():
    ug1 = -2.06
    uh = 6.3
    ih = 0.78

    assert abs(decode_ug1(encode_ug1(ug1)) - ug1) < 0.01
    assert abs(decode_uh(encode_uh(uh)) - uh) < 0.01
    assert abs(decode_ih(encode_ih(ih)) - ih) < 0.01


# ── get_param retry on name mismatch ─────────────────────────────────


class TestGetParamRetry:
    """get_param retries once after flushing when UART returns wrong param name."""

    def _make_client(self, monkeypatch):
        client = LM19Serial("COM_FAKE")
        monkeypatch.setattr(client, "flush_input", lambda: None)
        monkeypatch.setattr(client, "_write", lambda cmd: None)
        return client

    def test_retry_succeeds_on_name_mismatch(self, monkeypatch):
        """First response has garbage prefix ('0Ua'), retry returns correct name."""
        client = self._make_client(monkeypatch)
        responses = iter([
            ParamValue(name="0Ua", value=250),
            ParamValue(name="Ua", value=250),
        ])
        monkeypatch.setattr(client, "_read_param_response", lambda: next(responses))
        result = client.get_param("Ua", real=True)
        assert result == 250

    def test_raises_after_retry_still_mismatched(self, monkeypatch):
        """Both responses have wrong name — ValueError raised."""
        client = self._make_client(monkeypatch)
        responses = iter([
            ParamValue(name="0Ua", value=250),
            ParamValue(name="0Ua", value=250),
        ])
        monkeypatch.setattr(client, "_read_param_response", lambda: next(responses))
        with pytest.raises(ValueError, match="Unexpected param 0Ua"):
            client.get_param("Ua", real=True)

    def test_no_retry_when_name_matches(self, monkeypatch):
        """Normal response — no retry, single call to _read_param_response."""
        client = self._make_client(monkeypatch)
        call_count = [0]
        def read_once():
            call_count[0] += 1
            return ParamValue(name="Ua", value=200)
        monkeypatch.setattr(client, "_read_param_response", read_once)
        result = client.get_param("Ua")
        assert result == 200
        assert call_count[0] == 1
