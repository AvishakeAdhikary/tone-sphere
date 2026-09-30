"""
Opus, through ctypes against libopus — no Python binding.

The earlier blocker was packaging: no Python package shipped libopus for Windows, Linux and
macOS alike. So there is no package here at all. On Windows, libopus is built from Xiph's
pinned release by `scripts/build_native.py` into `opus.dll` beside the native engine (it is
BSD-licensed; the notice ships with it). On Linux and macOS it is the system's libopus
(`libopus0` from apt, `opus` from Homebrew), which CI installs. Where neither is present,
`available()` is False with the reason, and the Opus quality is refused with that reason —
never replaced by PCM.

Five calls are used: create, encode_float and destroy for the encoder, create,
decode_float and destroy for the decoder. Opus is stateful: an encoder belongs to one
stream and must see its frames in order, and a decoder must be handed packets in sequence
— or None for a lost one, which is Opus's own loss concealment. Encoding and decoding run
on network threads (the send worker, the playout thread), never on the audio thread.
"""

import ctypes
import ctypes.util
import platform
import sys
import threading
from pathlib import Path

import numpy as np

OPUS_OK = 0
OPUS_APPLICATION_AUDIO = 2049
OPUS_SET_BITRATE_REQUEST = 4002
RATES = (8000, 12000, 16000, 24000, 48000)
FRAME_MS = 10          # every packet ToneSphere sends is one 10 ms Opus frame
MAX_PACKET_BYTES = 1275 * 3
DEFAULT_BITRATE = 128_000

_lock = threading.Lock()
_lib: ctypes.CDLL | None = None
_reason: str | None = None


class OpusUnavailable(RuntimeError):
    """libopus is not loadable here, or cannot do what was asked."""


def _candidates() -> list[str]:
    if sys.platform == 'win32':
        return [str(Path(__file__).resolve().parents[1] / 'native' / '_bin' / 'opus.dll')]
    found = ctypes.util.find_library('opus')
    paths = [found] if found else []
    if sys.platform == 'darwin':
        paths += ['/opt/homebrew/lib/libopus.dylib', '/usr/local/lib/libopus.dylib']
    else:
        paths += ['libopus.so.0']
    return paths


def load() -> ctypes.CDLL:
    global _lib, _reason
    with _lock:
        if _lib is not None:
            return _lib
        errors = []
        for path in _candidates():
            try:
                lib = ctypes.CDLL(path)
                break
            except OSError as e:
                errors.append(f"{path}: {e}")
        else:
            _reason = ("libopus not found (" + ('; '.join(errors) or 'no candidate path') + "). On Windows build it "
                       "with scripts/build_native.py; on Linux install libopus0, on macOS `brew install opus`.")
            raise OpusUnavailable(_reason)
        c_int, c_void_p, c_float_p = ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)
        lib.opus_encoder_create.argtypes = [ctypes.c_int32, c_int, c_int, ctypes.POINTER(c_int)]
        lib.opus_encoder_create.restype = c_void_p
        lib.opus_encode_float.argtypes = [c_void_p, c_float_p, c_int, ctypes.c_char_p, ctypes.c_int32]
        lib.opus_encode_float.restype = ctypes.c_int32
        lib.opus_encoder_destroy.argtypes = [c_void_p]
        lib.opus_decoder_create.argtypes = [ctypes.c_int32, c_int, ctypes.POINTER(c_int)]
        lib.opus_decoder_create.restype = c_void_p
        lib.opus_decode_float.argtypes = [c_void_p, ctypes.c_char_p, ctypes.c_int32, c_float_p, c_int, c_int]
        lib.opus_decode_float.restype = c_int
        lib.opus_decoder_destroy.argtypes = [c_void_p]
        lib.opus_get_version_string.restype = ctypes.c_char_p
        lib.opus_strerror.argtypes = [c_int]
        lib.opus_strerror.restype = ctypes.c_char_p
        _lib = lib
        return lib


def available() -> bool:
    try:
        load()
        return True
    except OpusUnavailable:
        return False


def unavailable_reason() -> str | None:
    return None if available() else _reason


def version() -> str | None:
    return load().opus_get_version_string().decode() if available() else None


def frame_size(sample_rate: int) -> int:
    if sample_rate not in RATES:
        raise OpusUnavailable(f"Opus runs at {', '.join(f'{r // 1000} kHz' for r in RATES)}; "
                              f"this stream is at {sample_rate} Hz")
    return sample_rate * FRAME_MS // 1000


def _error(code: int) -> str:
    return load().opus_strerror(code).decode(errors='replace')


def _bitrate_settable() -> bool:
    # opus_encoder_ctl is variadic, and on Apple silicon variadic arguments are passed on
    # the stack, which a ctypes call with fixed argtypes does not do. There the encoder keeps
    # libopus's own default bitrate (about 100 kb/s for 48 kHz stereo), and says so.
    return not (sys.platform == 'darwin' and platform.machine() == 'arm64')


class Encoder:
    """One stream's encoder: 10 ms float32 frames in, one Opus packet each out."""

    def __init__(self, sample_rate: int, channels: int, bitrate: int = DEFAULT_BITRATE):
        self._handle = None   # before anything can raise, so __del__ always has it
        lib = load()
        self.frames = frame_size(sample_rate)
        self.channels = channels
        error = ctypes.c_int()
        self._handle = lib.opus_encoder_create(sample_rate, channels, OPUS_APPLICATION_AUDIO, ctypes.byref(error))
        if error.value != OPUS_OK or not self._handle:
            raise OpusUnavailable(f"opus_encoder_create: {_error(error.value)}")
        self.bitrate: int | None = None
        if _bitrate_settable():
            lib.opus_encoder_ctl.restype = ctypes.c_int
            if lib.opus_encoder_ctl(ctypes.c_void_p(self._handle), ctypes.c_int(OPUS_SET_BITRATE_REQUEST),
                                    ctypes.c_int32(bitrate)) == OPUS_OK:
                self.bitrate = bitrate
        self._out = ctypes.create_string_buffer(MAX_PACKET_BYTES)

    def encode(self, block: np.ndarray) -> bytes:
        pcm = np.ascontiguousarray(block, dtype=np.float32).reshape(-1, self.channels)
        if pcm.shape[0] != self.frames:
            raise ValueError(f"an Opus frame here is {self.frames} frames, not {pcm.shape[0]}")
        n = load().opus_encode_float(self._handle, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                     self.frames, self._out, MAX_PACKET_BYTES)
        if n < 0:
            raise OpusUnavailable(f"opus_encode_float: {_error(n)}")
        return self._out.raw[:n]

    def close(self):
        if self._handle:
            load().opus_encoder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class Decoder:
    """One stream's decoder. `decode(None)` conceals a lost packet with Opus's own PLC."""

    def __init__(self, sample_rate: int, channels: int):
        self._handle = None
        lib = load()
        self.frames = frame_size(sample_rate)
        self.channels = channels
        error = ctypes.c_int()
        self._handle = lib.opus_decoder_create(sample_rate, channels, ctypes.byref(error))
        if error.value != OPUS_OK or not self._handle:
            raise OpusUnavailable(f"opus_decoder_create: {_error(error.value)}")
        self._pcm = np.zeros((self.frames, channels), np.float32)

    def decode(self, packet: bytes | None) -> np.ndarray:
        n = load().opus_decode_float(self._handle, packet, len(packet) if packet else 0,
                                     self._pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), self.frames, 0)
        if n < 0:
            raise OpusUnavailable(f"opus_decode_float: {_error(n)}")
        return self._pcm[:n].copy()

    def close(self):
        if self._handle:
            load().opus_decoder_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
