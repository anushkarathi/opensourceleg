from typing import Protocol

import numpy as np

from opensourceleg.utilities import SoftRealtimeLoop


class Waveform(Protocol):
    def update(self, t: float) -> float: ...

    def __iter__(self) -> float: ...

    def __next__(self) -> float: ...


class SineWave:
    def __init__(self, amplitude: float, frequency: float, phase: float):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def update(self, t: float) -> float:
        return self.amplitude * np.sin(self.frequency * t + self.phase)

    def __iter__(self) -> float:
        return self

    def __next__(self) -> float:
        return self.update(self.t)


if __name__ == "__main__":
    frequency = 1000
    amplitude = 1
    phase = 0

    clock = SoftRealtimeLoop(dt=1 / frequency, maintain_original_phase=False)

    sw = SineWave(amplitude=amplitude, frequency=frequency, phase=phase)

    for t in clock:
        print(sw.update(t))
