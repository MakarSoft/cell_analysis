from dataclasses import dataclass
from typing import Tuple


@dataclass
class Measurement:
    """Измерение"""

    timestamp: str
    position: Tuple[float, float]  # (latitude, longitude)
    altitude: float
    rsrp: float
    rsrq: float

    def __post_init__(self):
        if not (-140 <= self.rsrp <= -40):
            raise ValueError(f"Invalid RSRP value: {self.rsrp}")
        if not (-20 <= self.rsrq <= 0):
            raise ValueError(f"Invalid RSRQ value: {self.rsrq}")
