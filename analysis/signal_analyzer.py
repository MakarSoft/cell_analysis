from typing import List, Tuple, Dict
from models.network_record import NetworkRecord
import numpy as np

class SignalAnalyzer:
    @staticmethod
    def analyze_band_performance(records: List[NetworkRecord]) -> Dict[str, Dict[str, float]]:
        band_data = {}
        for record in records:
            band = record.antenna.band
            if band not in band_data:
                band_data[band] = {
                    "rsrp_values": [],
                    "rsrq_values": []
                }
            
            band_data[band]["rsrp_values"].append(record.measurement.rsrp)
            band_data[band]["rsrq_values"].append(record.measurement.rsrq)
        
        results = {}
        for band, data in band_data.items():
            results[band] = {
                "mean_rsrp": np.mean(data["rsrp_values"]),
                "std_rsrp": np.std(data["rsrp_values"]),
                "mean_rsrq": np.mean(data["rsrq_values"]),
                "coverage_probability": np.mean(np.array(data["rsrp_values"]) > -100),
                "quality_probability": np.mean(np.array(data["rsrq_values"]) > -10)
            }
        
        return results