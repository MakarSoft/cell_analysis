from dataclasses import dataclass
from typing import Tuple


@dataclass
class Antenna:
    """Антена ..."""

    cellname: str
    band: str
    azimuth: float
    tilt: float
    hbw: float
    vbw: float
    position: Tuple[float, float]  # (latitude, longitude)
    height: float

    def __post_init__(self):
        if self.band not in ["LTE1800", "LTE2100"]:
            raise ValueError(f"Invalid band: {self.band}")

        # Азимут всегда положительный:
        #   Измеряется по часовой стрелке от севера (0°)
        #       Диапазон: 0° до 360°
        #   Типичные значения:
        #       0° = север
        #       90° = восток
        #       180° = юг
        #       270° = запад
        #       240° = юго-запад

        # Нормализация азимута в диапазон [0, 360)
        self.azimuth = self.azimuth % 360

        # Проверка угла наклона (tilt)
        # Обычно в диапазоне -15° до +15°
        if not (-15 <= self.tilt <= 15):
            raise ValueError(
                f"Invalid tilt: {self.tilt}. Expected between -15° and 15°"
            )

        # Проверка ширины луча
        # hbw: Горизонтальная ширина луча (1-360°)
        # vbw: Вертикальная ширина луча (1-180°)
        if not (0 < self.hbw <= 360) or not (0 < self.vbw <= 180):
            raise ValueError(
                f"Invalid beamwidth: HBW={self.hbw}, VBW={self.vbw}"
            )

    def __hash__(self):
        return hash((
            self.cellname,
            self.band,
            self.azimuth,
            self.tilt,
            self.hbw,
            self.vbw,
            self.position[0],
            self.position[1],
            self.height
        ))

    def __eq__(self, other):
        if not isinstance(other, Antenna):
            return False
        return (
            self.cellname == other.cellname and
            self.band == other.band and
            self.azimuth == other.azimuth and
            self.tilt == other.tilt and
            self.hbw == other.hbw and
            self.vbw == other.vbw and
            self.position == other.position and
            self.height == other.height
        )
