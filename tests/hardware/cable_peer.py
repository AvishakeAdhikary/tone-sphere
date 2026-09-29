"""
Another application, for the virtual driver tests: a separate process, using PortAudio
(through sounddevice) rather than ToneSphere's native engine, so audio crossing the virtual
cable is proven to cross between applications through Windows, not within ToneSphere.

    python cable_peer.py play   <device name part> <seconds> <frequency> <amplitude>
    python cable_peer.py record <device name part> <seconds> <output.npy>
"""

import sys

import numpy as np
import sounddevice as sd

RATE = 48000


def find(name_part: str, output: bool) -> int:
    apis = sd.query_hostapis()
    for i, d in enumerate(sd.query_devices()):
        if 'WASAPI' not in apis[d['hostapi']]['name'] or name_part.lower() not in d['name'].lower():
            continue
        if (d['max_output_channels'] if output else d['max_input_channels']) > 0:
            return i
    raise SystemExit(f"no WASAPI {'output' if output else 'input'} matching {name_part!r}")


def main() -> int:
    mode, name, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
    if mode == 'play':
        freq, amplitude = float(sys.argv[4]), float(sys.argv[5])
        t = np.arange(int(RATE * seconds)) / RATE
        tone = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(np.column_stack([tone, tone]), RATE, device=find(name, output=True), blocking=True)
    else:
        audio = sd.rec(int(RATE * seconds), RATE, channels=2, dtype='float32', device=find(name, output=False),
                       blocking=True)
        np.save(sys.argv[4], audio)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
