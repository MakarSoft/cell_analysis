from geopy.distance import geodesic
import numpy as np
from typing import List, Dict, Any
from models.network_record import NetworkRecord


class SpatialCalculator:
    @staticmethod
    def calculate_metrics(
        records: List[NetworkRecord]
    ) -> List[Dict[str, Any]]:

        results = []
        for record in records:
            antenna_pos = record.antenna.position
            point_pos = record.measurement.position

            # Расстояние в метрах
            distance = geodesic(antenna_pos, point_pos).meters

            # Расчет угла от антенны
            dx = point_pos[1] - antenna_pos[1]
            dy = point_pos[0] - antenna_pos[0]

            angle = np.degrees(np.arctan2(dx, dy)) % 360
            
            azimuth_diff = (record.antenna.azimuth - angle) % 360
            if azimuth_diff > 180:
                azimuth_diff -= 360

            results.append({
                "record": record,
                "distance": distance,
                "azimuth_diff": azimuth_diff
            })
        return results
