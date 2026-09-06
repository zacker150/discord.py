# -*- coding: utf-8 -*-

"""

Tests for the DAVE E2EE plumbing on the voice websocket

"""

import asyncio
import inspect
import struct
import threading

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
    assert DiscordVoiceWebSocket.FLAGS == 18
    assert DiscordVoiceWebSocket.PLATFORM == 20
