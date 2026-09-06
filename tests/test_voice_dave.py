# -*- coding: utf-8 -*-

"""

Tests for the DAVE E2EE plumbing on the voice websocket

"""

import asyncio
import inspect
import struct
import threading
import types

import pytest

from discord.gateway import DiscordVoiceWebSocket
from discord.utils import MISSING
from discord.voice_state import ConnectionFlowState, VoiceConnectionState


class StubDaveSession:
    def __init__(self):
        self.external_sender = None
        self.ready = False

    def set_external_sender(self, payload):
        self.external_sender = payload


class StubConnectionState:
    def __init__(self, dave_session=None):
        self.dave_session = dave_session
        self.dave_protocol_version = 1
        self.dave_pending_transitions = {}


def make_ws(state, binary_hook=None):
    ws = DiscordVoiceWebSocket(None, None, binary_hook=binary_hook)  # type: ignore
    ws._connection = state  # type: ignore
    return ws


def binary_frame(seq: int, op: int, payload: bytes) -> bytes:
    return struct.pack('>H', seq) + bytes([op]) + payload


@pytest.mark.asyncio
async def test_binary_hook_called_without_session():
    # A frame that arrives before the group exists must still reach the hook,
    # otherwise an extension cannot observe the start of the handshake.
    calls = []

    async def hook(ws, op, seq, payload):
        calls.append((ws, op, seq, payload))

    state = StubConnectionState(dave_session=None)
    ws = make_ws(state, binary_hook=hook)

    await ws.received_binary_message(binary_frame(7, DiscordVoiceWebSocket.MLS_EXTERNAL_SENDER, b'sender'))

    assert calls == [(ws, DiscordVoiceWebSocket.MLS_EXTERNAL_SENDER, 7, b'sender')]
    assert ws.seq_ack == 7


@pytest.mark.asyncio
async def test_binary_hook_called_after_session_is_updated():
    calls = []
    session = StubDaveSession()

    async def hook(ws, op, seq, payload):
        calls.append((op, seq, payload, session.external_sender))

    state = StubConnectionState(dave_session=session)
    ws = make_ws(state, binary_hook=hook)

    await ws.received_binary_message(binary_frame(3, DiscordVoiceWebSocket.MLS_EXTERNAL_SENDER, b'sender'))

    # The hook runs last, so the session it inspects is already up to date.
    assert calls == [(DiscordVoiceWebSocket.MLS_EXTERNAL_SENDER, 3, b'sender', b'sender')]
    assert session.external_sender == b'sender'


@pytest.mark.asyncio
async def test_default_binary_hook_is_a_no_op():
    state = StubConnectionState(dave_session=StubDaveSession())
    ws = make_ws(state)

    await ws.received_binary_message(binary_frame(1, DiscordVoiceWebSocket.MLS_EXTERNAL_SENDER, b'sender'))

    assert ws.seq_ack == 1


@pytest.mark.asyncio
async def test_connect_websocket_forwards_binary_hook(monkeypatch):
    async def hook(*args):
        pass

    async def binary_hook(*args):
        pass

    received = {}

    async def from_connection_state(cls, state, **kwargs):
        received.update(kwargs)
        return 'ws'

    monkeypatch.setattr(
        DiscordVoiceWebSocket,
        'from_connection_state',
        classmethod(from_connection_state),
    )

    state = VoiceConnectionState.__new__(VoiceConnectionState)
    state.hook = hook  # type: ignore
    state.binary_hook = binary_hook  # type: ignore
    state.ws = MISSING
    state._state = ConnectionFlowState.disconnected
    state._state_event = asyncio.Event()
    state._connected = threading.Event()

    await VoiceConnectionState._connect_websocket(state, False)

    assert received['hook'] is hook
    assert received['binary_hook'] is binary_hook


def test_voice_connection_state_accepts_binary_hook():
    # The voice-recv extension feature-detects the fork with exactly this check.
    assert 'binary_hook' in inspect.signature(VoiceConnectionState.__init__).parameters


def test_voice_opcodes():
    assert DiscordVoiceWebSocket.CLIENTS_CONNECT == 11
    assert DiscordVoiceWebSocket.CLIENT_CONNECT == 12
    assert DiscordVoiceWebSocket.CLIENT_DISCONNECT == 13
    assert DiscordVoiceWebSocket.MEDIA_SINK_WANTS == 15
    assert DiscordVoiceWebSocket.FLAGS == 18
    assert DiscordVoiceWebSocket.PLATFORM == 20


# --- A2: received_message restructure -------------------------------------


class ScriptedDaveSession:
    """Records what the handshake asks of davey."""

    instances = []

    def __init__(self, protocol_version, user_id, channel_id):
        self.protocol_version = protocol_version
        self.user_id = user_id
        self.channel_id = channel_id
        self.ready = True
        self.epoch = 1
        self.passthrough_calls = []
        self.reinit_calls = []
        self.reset_calls = 0
        ScriptedDaveSession.instances.append(self)

    def reinit(self, protocol_version, user_id, channel_id):
        self.protocol_version = protocol_version
        self.reinit_calls.append((protocol_version, user_id, channel_id))

    def reset(self):
        self.reset_calls += 1

    def set_passthrough_mode(self, enabled, duration):
        self.passthrough_calls.append((enabled, duration))

    def get_serialized_key_package(self):
        return b'key-package'


class RecordingVoiceClient:
    def __init__(self):
        self.channel = None
        self.ws = None
        self.user = StubUser()
        self.supported_modes = ('aead_xchacha20_poly1305_rtpsize',)
        self.dave_events = []

    def on_dave_transition_prepared(self, transition_id, protocol_version):
        self.dave_events.append(('prepared', transition_id, protocol_version))

    def on_dave_transition_executed(self, transition_id, protocol_version):
        self.dave_events.append(('executed', transition_id, protocol_version))

    def on_dave_epoch_prepared(self, epoch, protocol_version):
        self.dave_events.append(('epoch', epoch, protocol_version))


class StubChannel:
    def __init__(self, channel_id=999, voice_states=None):
        self.id = channel_id
        self.voice_states = voice_states or {}


class StubUser:
    def __init__(self, user_id=42):
        self.id = user_id


@pytest.fixture
def harness(monkeypatch):
    """A real VoiceConnectionState and websocket wired to stubs."""
    import discord.voice_state as voice_state

    ScriptedDaveSession.instances = []
    # Substitute the whole module rather than patching the real one, so these
    # tests run whether or not davey is installed.
    monkeypatch.setattr(voice_state, 'davey', types.SimpleNamespace(DaveSession=ScriptedDaveSession), raising=False)
    monkeypatch.setattr(voice_state, 'has_dave', True)

    voice_client = RecordingVoiceClient()
    voice_client.channel = StubChannel()

    state = VoiceConnectionState.__new__(VoiceConnectionState)
    state.voice_client = voice_client  # type: ignore
    state.dave_session = None
    state.dave_protocol_version = 0
    state.dave_pending_transitions = {}
    state.dave_downgraded = False
    state.dave_known_user_ids = set()

    ws = DiscordVoiceWebSocket(None, None)  # type: ignore
    ws._connection = state  # type: ignore
    voice_client.ws = ws

    sent_json = []
    sent_binary = []

    async def send_as_json(payload):
        sent_json.append(payload)

    async def send_binary(opcode, payload):
        sent_binary.append((opcode, payload))

    ws.send_as_json = send_as_json  # type: ignore
    ws.send_binary = send_binary  # type: ignore

    return state, ws, sent_json, sent_binary, voice_client


def frame(op, data):
    return {'op': op, 'd': data}


@pytest.mark.asyncio
async def test_prepare_epoch_creates_group_without_existing_session(harness):
    # G1: a call that starts at version 0 and upgrades via op 24 must still join.
    state, ws, sent_json, sent_binary, vc = harness

    await ws.received_message(frame(DiscordVoiceWebSocket.DAVE_PREPARE_EPOCH, {'epoch': 1, 'protocol_version': 1}))

    assert state.dave_protocol_version == 1
    assert state.dave_session is not None
    assert sent_binary == [(DiscordVoiceWebSocket.MLS_KEY_PACKAGE, b'key-package')]
    assert vc.dave_events == [('epoch', 1, 1)]


@pytest.mark.asyncio
async def test_prepare_transition_upgrade_without_existing_session(harness):
    # G1 again, via the op 21 path: reinit, then acknowledge.
    state, ws, sent_json, sent_binary, vc = harness

    await ws.received_message(
        frame(DiscordVoiceWebSocket.DAVE_PREPARE_TRANSITION, {'transition_id': 7, 'protocol_version': 1})
    )

    assert state.dave_session is not None
    assert sent_binary == [(DiscordVoiceWebSocket.MLS_KEY_PACKAGE, b'key-package')]
    assert sent_json == [{'op': DiscordVoiceWebSocket.DAVE_TRANSITION_READY, 'd': {'transition_id': 7}}]
    assert state.dave_pending_transitions == {7: 1}
    assert vc.dave_events == [('prepared', 7, 1)]


@pytest.mark.asyncio
async def test_downgrade_sets_passthrough_then_executes(harness):
    state, ws, sent_json, sent_binary, vc = harness
    state.dave_protocol_version = 1
    session = ScriptedDaveSession(1, 42, 999)
    state.dave_session = session

    await ws.received_message(
        frame(DiscordVoiceWebSocket.DAVE_PREPARE_TRANSITION, {'transition_id': 3, 'protocol_version': 0})
    )
    assert session.passthrough_calls == [(True, 120)]
    assert sent_json == [{'op': DiscordVoiceWebSocket.DAVE_TRANSITION_READY, 'd': {'transition_id': 3}}]

    await ws.received_message(frame(DiscordVoiceWebSocket.DAVE_EXECUTE_TRANSITION, {'transition_id': 3}))

    assert state.dave_protocol_version == 0
    assert state.dave_downgraded is True
    assert vc.dave_events == [('prepared', 3, 0), ('executed', 3, 0)]


@pytest.mark.asyncio
@pytest.mark.parametrize('protocol_version', [0, 1])
async def test_prepare_transition_id_zero_executes_immediately(harness, protocol_version):
    state, ws, sent_json, sent_binary, vc = harness
    state.dave_protocol_version = 1
    state.dave_session = ScriptedDaveSession(1, 42, 999)

    await ws.received_message(
        frame(DiscordVoiceWebSocket.DAVE_PREPARE_TRANSITION, {'transition_id': 0, 'protocol_version': protocol_version})
    )

    # Executed inline, so no transition-ready ack and nothing left pending.
    assert sent_json == []
    assert state.dave_pending_transitions == {}
    assert state.dave_protocol_version == protocol_version
    assert vc.dave_events == [('prepared', 0, protocol_version), ('executed', 0, protocol_version)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('callback_name', 'op', 'data'),
    [
        ('on_dave_transition_prepared', 21, {'transition_id': 7, 'protocol_version': 0}),
        ('on_dave_transition_prepared', 21, {'transition_id': 0, 'protocol_version': 0}),
        ('on_dave_transition_executed', 21, {'transition_id': 0, 'protocol_version': 0}),
        ('on_dave_transition_executed', 22, {'transition_id': 7}),
        ('on_dave_epoch_prepared', 24, {'epoch': 1, 'protocol_version': 1}),
    ],
)
async def test_callback_failure_does_not_disrupt_protocol(harness, caplog, callback_name, op, data):
    state, ws, sent_json, sent_binary, vc = harness
    state.dave_protocol_version = 1
    session = ScriptedDaveSession(1, 42, 999)
    state.dave_session = session
    if op == DiscordVoiceWebSocket.DAVE_EXECUTE_TRANSITION:
        state.dave_pending_transitions[7] = 0

    def boom(*args):
        raise RuntimeError('application callback failed')

    setattr(vc, callback_name, boom)
    await ws.received_message(frame(op, data))

    assert 'application callback failed' in caplog.text
    if op == DiscordVoiceWebSocket.DAVE_PREPARE_EPOCH:
        assert sent_json == []
        assert sent_binary == [(DiscordVoiceWebSocket.MLS_KEY_PACKAGE, b'key-package')]
        assert session.reinit_calls == [(1, 42, 999)]
    else:
        assert sent_binary == []
        assert session.reinit_calls == []
        assert session.reset_calls == 0
        if op == DiscordVoiceWebSocket.DAVE_PREPARE_TRANSITION and data['transition_id'] != 0:
            assert sent_json == [{'op': DiscordVoiceWebSocket.DAVE_TRANSITION_READY, 'd': {'transition_id': 7}}]
            assert state.dave_pending_transitions == {7: 0}
        else:
            assert sent_json == []
            assert state.dave_pending_transitions == {}
            assert state.dave_protocol_version == 0
            assert state.dave_downgraded is True
            if callback_name == 'on_dave_transition_prepared':
                assert vc.dave_events == [('executed', 0, 0)]


@pytest.mark.asyncio
async def test_dave_handler_error_does_not_escape_and_rekeys(harness):
    # G2: a davey failure must not kill the poller.
    state, ws, sent_json, sent_binary, vc = harness
    state.dave_protocol_version = 1
    session = ScriptedDaveSession(1, 42, 999)
    state.dave_session = session

    def boom(enabled, duration):
        raise ValueError('davey exploded')

    session.set_passthrough_mode = boom

    await ws.received_message(
        frame(DiscordVoiceWebSocket.DAVE_PREPARE_TRANSITION, {'transition_id': 4, 'protocol_version': 0})
    )

    assert sent_json == [{'op': DiscordVoiceWebSocket.MLS_INVALID_COMMIT_WELCOME, 'd': {'transition_id': 4}}]


@pytest.mark.asyncio
async def test_membership_tracking(harness):
    state, ws, sent_json, sent_binary, vc = harness

    await ws.received_message(frame(DiscordVoiceWebSocket.CLIENTS_CONNECT, {'user_ids': ['10', '11']}))
    assert state.dave_known_user_ids == {10, 11}

    await ws.received_message(frame(DiscordVoiceWebSocket.SPEAKING, {'user_id': '12', 'ssrc': 1, 'speaking': 1}))
    assert state.dave_known_user_ids == {10, 11, 12}

    await ws.received_message(frame(DiscordVoiceWebSocket.CLIENT_DISCONNECT, {'user_id': '11'}))
    assert state.dave_known_user_ids == {10, 12}


@pytest.mark.asyncio
async def test_membership_tracking_survives_unexpected_payload(harness):
    state, ws, sent_json, sent_binary, vc = harness

    await ws.received_message(frame(DiscordVoiceWebSocket.CLIENT_DISCONNECT, {}))

    assert state.dave_known_user_ids == set()


@pytest.mark.asyncio
@pytest.mark.parametrize('next_user_ids', [(12, 42), (42,), (), None])
async def test_membership_seeded_from_channel_voice_states(harness, next_user_ids):
    state, ws, sent_json, sent_binary, vc = harness
    vc.channel = StubChannel(voice_states={10: object(), 11: object(), 42: object()})

    class StubLoop:
        async def sock_connect(self, sock, addr):
            pass

    ws.loop = StubLoop()  # type: ignore
    state.socket = object()  # type: ignore

    async def discover_ip():
        return ('1.2.3.4', 5000)

    async def select_protocol(ip, port, mode):
        pass

    ws.discover_ip = discover_ip  # type: ignore
    ws.select_protocol = select_protocol  # type: ignore

    await ws.initial_connection({'ssrc': 1, 'port': 2, 'ip': '1.2.3.4', 'modes': ['aead_xchacha20_poly1305_rtpsize']})

    # Our own id is excluded; the proposal check adds it back.
    assert state.dave_known_user_ids == {10, 11}

    # A fresh READY after a reconnect or channel move replaces the old snapshot,
    # including when the new channel has an empty or unavailable cache.
    vc.channel = StubChannel(channel_id=1000, voice_states=dict.fromkeys(next_user_ids or ()))
    if next_user_ids is None:
        del vc.channel.voice_states
    await ws.initial_connection({'ssrc': 2, 'port': 3, 'ip': '1.2.3.4', 'modes': ['aead_xchacha20_poly1305_rtpsize']})
    assert state.dave_known_user_ids == set(next_user_ids or ()) - {42}
