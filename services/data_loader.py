import pandas as pd
from typing import List
from models.network_record import NetworkRecord
from models.antenna import Antenna
from models.measurement import Measurement


class DataLoader:
    """
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def load_data(self) -> List[NetworkRecord]:
        """Загрузка данных из CSV"""

        df = pd.read_csv(self.file_path)
        records = []

        for _, row in df.iterrows():
            try:
                antenna = Antenna(
                    cellname=row['cellname'],
                    band=row['band'],
                    azimuth=row['azimuth'],
                    tilt=row['tilt'],
                    hbw=row['hbw'],
                    vbw=row['vbw'],
                    position=(row['nc_latitude'], row['nc_longitude']),
                    height=row['height']
                )

                measurement = Measurement(
                    timestamp=row['eventtime'],
                    position=(row['latitude'], row['longitude']),
                    altitude=row['altitude'],
                    rsrp=row['servingcellrsrp'],
                    rsrq=row['servingcellrsrq']
                )

                records.append(NetworkRecord(antenna, measurement))
            except ValueError as e:
                print(f"Skipping invalid record: {e}")

        return records
