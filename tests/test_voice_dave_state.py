import asyncio
import threading
import types
from unittest.mock import AsyncMock, Mock

import pytest

from discord.voice_client import VoiceClient
from discord.errors import ConnectionClosed
from discord.voice_state import ConnectionFlowState, VoiceConnectionState
from discord.utils import MISSING
from test_voice_dave import harness, binary_harness, binary_frame, frame, ScriptedDaveSession


def seed(state):
    state.dave_session = ScriptedDaveSession(1, 42, 999)
    state.dave_protocol_version = 1
    state.dave_pending_transitions = {7: 1}
    state.dave_downgraded = True
    state.dave_known_user_ids = {10}
    return state.dave_session


def assert_reset(state):
    assert state.dave_session is None
    assert state.dave_protocol_version == 0
    assert state.dave_pending_transitions == {}
    assert state.dave_downgraded is False
    assert state.dave_known_user_ids == set()


def available_to_worker(lock):
    results = []

    def worker():
        acquired = lock.acquire(blocking=False)
        results.append(acquired)
        if acquired:
            lock.release()

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    return results == [True]


@pytest.mark.parametrize('raises', [False, True])
def test_reset_clears_all_state_even_when_native_reset_fails(harness, raises):
    state, ws, _, _, _ = harness
    session = seed(state)

    def reset():
        assert not available_to_worker(state.dave_lock)
        if raises:
            raise ValueError('reset failed')

    session.reset = reset
    generation = state.dave_session_generation
    state._reset_dave_state()
    assert state.dave_session_generation == generation + 1
    assert_reset(state)
    assert available_to_worker(state.dave_lock)


@pytest.mark.asyncio
@pytest.mark.parametrize('transition_id', [0, 7])
async def test_upgrade_expires_passthrough(harness, transition_id, caplog):
    state, ws, _, _, _ = harness
    session = seed(state)
    state.dave_protocol_version = 0
    state.dave_pending_transitions = {transition_id: 1}
    passthrough = session.set_passthrough_mode

    def locked_passthrough(*args):
        assert not available_to_worker(state.dave_lock)
        passthrough(*args)

    session.set_passthrough_mode = locked_passthrough
    with caplog.at_level('INFO'):
        await state._execute_transition(transition_id)
    assert session.passthrough_calls == [(False, 10)]
    assert state.dave_protocol_version == 1
    assert not state.dave_downgraded
    assert 'upgraded' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('missing', ['ws', 'channel'])
async def test_reinit_without_connection_preserves_state(harness, missing):
    state, ws, sent_json, sent_binary, vc = harness
    session = seed(state)
    if missing == 'ws':
        state.ws = MISSING
    else:
        vc.channel = None
    await state.reinit_dave_session()
    await state._recover_from_invalid_commit(7)
    assert session.reinit_calls == []
    assert state.dave_session is session
    assert sent_binary == sent_json == []
    assert state.dave_session_generation == 0


@pytest.mark.asyncio
async def test_reinit_native_calls_locked_but_send_unlocked(harness):
    state, ws, _, _, vc = harness
    session = seed(state)
    vc.channel.id = 1000

    def reinit(*args):
        assert args == (1, 42, 1000)
        assert state.dave_session_generation == 1
        assert not available_to_worker(state.dave_lock)

    def key_package():
        assert not available_to_worker(state.dave_lock)
        return b'key'

    async def send(op, body):
        assert available_to_worker(state.dave_lock)
        assert body == b'key'

    session.reinit = reinit
    session.get_serialized_key_package = key_package
    ws.send_binary = send
    await state.reinit_dave_session()
    assert state.dave_session is session


@pytest.mark.asyncio
async def test_version_zero_reinit_keeps_passthrough(harness):
    state, ws, _, sent_binary, _ = harness
    session = seed(state)
    state.dave_protocol_version = 0
    await state.reinit_dave_session()
    assert session.reset_calls == 1
    assert session.passthrough_calls == [(True, 10)]
    assert sent_binary == []


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup', [False, True])
async def test_disconnect_cleanup_controls_reset_even_on_transport_failure(harness, cleanup):
    state, ws, _, _, vc = harness
    session = seed(state)
    state._state_event = asyncio.Event()
    state._connected = threading.Event()
    state._state = ConnectionFlowState.connected
    state._socket_reader = Mock()
    state.socket = Mock()
    state._voice_disconnect = AsyncMock(side_effect=ConnectionError('disconnect failed'))
    vc.stop = Mock()
    vc.cleanup = Mock()
    await state.disconnect(cleanup=cleanup)
    if cleanup:
        assert_reset(state)
        assert session.reset_calls == 1
        vc.cleanup.assert_called_once()
    else:
        assert state.dave_session is session
        assert session.reset_calls == 0
        vc.cleanup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('resume', [False, True])
async def test_connection_reset_precedes_handshake_but_resume_preserves_session(harness, resume):
    state, ws, _, _, _ = harness
    session = seed(state)

    async def connect(**kwargs):
        if resume:
            assert state.dave_session is session
        else:
            assert_reset(state)
        raise asyncio.CancelledError()

    state._voice_connect = connect
    with pytest.raises(asyncio.CancelledError):
        await state._inner_connect(True, False, False, resume)
    assert session.reset_calls == (0 if resume else 1)


@pytest.mark.asyncio
async def test_channel_move_resets_before_new_websocket(harness):
    state, old_ws, _, _, _ = harness
    session = seed(state)
    state._state = ConnectionFlowState.got_both_voice_updates
    state.timeout = 1
    state._wait_for_state = AsyncMock()
    old_ws.close = AsyncMock()
    new_ws = Mock()

    async def connect(resume):
        assert resume is False
        assert_reset(state)
        return new_ws

    state._connect_websocket = connect
    state._handshake_websocket = AsyncMock()
    assert await state._potential_reconnect()
    assert session.reset_calls == 1
    assert state.ws is new_ws
    old_ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_poller_logs_backs_off_and_processes_next_event(harness, monkeypatch, caplog):
    state, ws, _, _, _ = harness
    ws.poll_event = AsyncMock(side_effect=[RuntimeError('handler failed'), None, asyncio.CancelledError()])
    sleep = AsyncMock()
    monkeypatch.setattr('discord.voice_state.asyncio.sleep', sleep)
    await state._poll_voice_ws(True)
    assert ws.poll_event.await_count == 3
    sleep.assert_awaited_once_with(1.0)
    assert 'handler failed' in caplog.text


@pytest.mark.asyncio
async def test_4015_poller_uses_resume_without_resetting_session(harness):
    state, ws, _, _, vc = harness
    session = seed(state)
    state.timeout = 1
    state.self_deaf = state.self_mute = False
    vc.guild = types.SimpleNamespace(me=types.SimpleNamespace(voice=None))
    ws.poll_event = AsyncMock(side_effect=[ConnectionClosed(Mock(), shard_id=None, code=4015), asyncio.CancelledError()])
    state._connect = AsyncMock()
    await state._poll_voice_ws(True)
    assert state._connect.await_args.kwargs['resume'] is True
    assert state.dave_session is session
    assert session.reset_calls == 0


@pytest.mark.asyncio
async def test_poller_backoff_preserves_cancellation(harness, monkeypatch):
    state, ws, _, _, _ = harness
    ws.poll_event = AsyncMock(side_effect=RuntimeError('handler failed'))
    monkeypatch.setattr('discord.voice_state.asyncio.sleep', AsyncMock(side_effect=asyncio.CancelledError()))
    with pytest.raises(asyncio.CancelledError):
        await state._poll_voice_ws(True)
    ws.poll_event.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'op, method, payload',
    [
        (25, 'set_external_sender', b'sender'),
        (27, 'process_proposals', b'\x00proposals'),
        (29, 'process_commit', b'\x00\x07commit'),
        (30, 'process_welcome', b'\x00\x07welcome'),
    ],
)
async def test_binary_native_calls_exclude_workers_and_release_for_hooks(binary_harness, op, method, payload, caplog):
    harness, session = binary_harness
    state, ws, _, _, vc = harness

    def native(*args, **kwargs):
        assert not available_to_worker(state.dave_lock)

    getattr(session, method).side_effect = native

    async def hook(*args):
        assert available_to_worker(state.dave_lock)

    ws._binary_hook = hook
    ws.send_transition_ready = hook
    await ws.received_binary_message(binary_frame(1, op, payload))
    getattr(session, method).assert_called_once()
    assert available_to_worker(state.dave_lock)
    assert not [record for record in caplog.records if record.levelno >= 40]


@pytest.mark.asyncio
async def test_json_passthrough_locked_and_callbacks_unlocked(harness, caplog):
    state, ws, _, _, vc = harness
    session = seed(state)

    def passthrough(*args):
        assert not available_to_worker(state.dave_lock)

    def callback(*args):
        assert available_to_worker(state.dave_lock)

    session.set_passthrough_mode = passthrough
    vc.on_dave_transition_prepared = callback
    vc.on_dave_transition_executed = callback
    await ws.received_message(frame(ws.DAVE_PREPARE_TRANSITION, {'transition_id': 0, 'protocol_version': 0}))
    assert state.dave_protocol_version == 0
    assert not [record for record in caplog.records if record.levelno >= 40]


def test_player_encrypts_under_shared_lock(harness):
    state, ws, _, _, _ = harness
    session = seed(state)

    def encrypt(data):
        assert not available_to_worker(state.dave_lock)
        return b'ciphertext'

    session.encrypt_opus = encrypt
    client = VoiceClient.__new__(VoiceClient)
    client._connection = state
    client.sequence = client.timestamp = 0
    state.ssrc = 1
    state.mode = 'test'

    def transport(header, packet):
        assert available_to_worker(state.dave_lock)
        return packet

    client._encrypt_test = transport
    assert client.dave_lock is state.dave_lock
    assert client._get_voice_packet(b'opus') == b'ciphertext'
