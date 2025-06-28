from dataclasses import dataclass
from .antenna import Antenna
from .measurement import Measurement


@dataclass
class NetworkRecord:
    antenna: Antenna
    measurement: Measurement
