import subprocess
import sys
import types
from unittest.mock import Mock

import pytest

from discord.voice_client import VoiceClient
from test_voice_dave import harness, ScriptedDaveSession
from test_voice_dave_state import available_to_worker


@pytest.fixture
def client(harness):
    state, _, _, _, _ = harness
    client = VoiceClient.__new__(VoiceClient)
    client._connection = state
    client.sequence = client.timestamp = 0
    client._dave_frames_dropped = 0
    client._dave_waiting_for_ready = False
    client.encoder = types.SimpleNamespace(SAMPLES_PER_FRAME=960, encode=Mock(return_value=b'encoded'))
    client._encrypt_test = Mock(side_effect=lambda header, data: bytes(header) + data)
    state.mode = 'test'
    state.ssrc = 1
    state.send_packet = Mock()
    return client


def session_for(client, ready=True):
    session = ScriptedDaveSession(1, 42, 999)
    session.ready = ready
    session.encrypt_opus = Mock(return_value=b'ciphertext')
    session.get_verification_code = Mock(return_value='verification-code')
    client._connection.dave_session = session
    client._connection.dave_protocol_version = 1
    return session


@pytest.mark.parametrize('version', [0, 1])
@pytest.mark.parametrize('readiness', [None, False, True])
def test_packet_readiness_and_plaintext_policy(client, version, readiness):
    session = session_for(client, readiness) if readiness is not None else None
    client._connection.dave_protocol_version = version
    packet = client._get_voice_packet(b'opus')
    if version > 0 and not readiness:
        assert packet is None
        client._encrypt_test.assert_not_called()
    else:
        assert packet[12:] == (b'ciphertext' if version else b'opus')
    if session:
        assert session.encrypt_opus.call_count == (1 if version and readiness else 0)


@pytest.mark.parametrize('encode', [False, True])
def test_drops_count_and_log_once_per_gap_then_resume(client, encode, caplog):
    session = session_for(client, False)
    for _ in range(3):
        client.send_audio_packet(b'audio', encode=encode)
    assert client._dave_frames_dropped == 3
    assert client.timestamp == 3 * 960
    client._connection.send_packet.assert_not_called()
    assert len(caplog.records) == 1

    session.ready = True
    client.send_audio_packet(b'audio', encode=encode)
    assert client._connection.send_packet.call_count == 1
    assert client._dave_frames_dropped == 3
    assert client.timestamp == 4 * 960
    session.encrypt_opus.assert_called_once_with(b'encoded' if encode else b'audio')

    session.ready = False
    client.send_audio_packet(b'audio', encode=encode)
    assert client._dave_frames_dropped == 4
    assert len(caplog.records) == 2
    assert client.timestamp == 5 * 960
    assert client._connection.send_packet.call_count == 1


def test_native_encryption_error_never_falls_back_to_plaintext(client):
    session = session_for(client)
    session.encrypt_opus.side_effect = ValueError('native encryption failed')
    with pytest.raises(ValueError, match='native encryption failed'):
        client.send_audio_packet(b'opus', encode=False)
    client._connection.send_packet.assert_not_called()
    client._encrypt_test.assert_not_called()


def test_readiness_api_before_join_when_ready_and_after_cleanup(client):
    assert client.dave_protocol_version == 0
    assert not client.dave_ready
    assert client.dave_epoch is None
    assert client.get_dave_verification_code(10) is None
    session = session_for(client, False)
    session.epoch = None
    assert not client.dave_ready
    assert client.dave_epoch is None
    assert client.get_dave_verification_code(10) is None
    session.get_verification_code.assert_not_called()

    session.ready = True
    session.epoch = 3
    assert client.dave_protocol_version == 1
    assert client.dave_ready
    assert client.dave_epoch == 3
    assert client.get_dave_verification_code(10) == 'verification-code'
    session.get_verification_code.assert_called_once_with(10)

    client._connection.dave_protocol_version = 0
    assert not client.dave_ready
    assert client.get_dave_verification_code(10) is None
    client._connection._reset_dave_state()
    assert client.dave_protocol_version == 0
    assert not client.dave_ready
    assert client.dave_epoch is None
    assert client.get_dave_verification_code(10) is None


def test_verification_lookup_locked_and_native_errors_propagate(client):
    session = session_for(client)

    def lookup(user_id):
        assert not available_to_worker(client.dave_lock)
        raise ValueError('unknown member')

    session.get_verification_code.side_effect = lookup
    with pytest.raises(ValueError, match='unknown member'):
        client.get_dave_verification_code(10)
    assert available_to_worker(client.dave_lock)


@pytest.mark.parametrize('gil', ['enabled', 'disabled', 'unavailable'])
def test_free_threaded_import_warning(gil):
    setup = {
        'enabled': 'sys._is_gil_enabled = lambda: True',
        'disabled': 'sys._is_gil_enabled = lambda: False',
        'unavailable': "if hasattr(sys, '_is_gil_enabled'): del sys._is_gil_enabled",
    }[gil]
    result = subprocess.run(
        [sys.executable, '-c', f'import sys\n{setup}\nimport discord.voice_client'],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert ('DAVE voice is unvalidated on free-threaded Python' in result.stderr) == (gil == 'disabled')
