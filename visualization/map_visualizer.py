import folium
from folium.plugins import HeatMap, MarkerCluster
from typing import List, Tuple
from models.network_record import NetworkRecord
from collections import defaultdict


class MapVisualizer:
    def __init__(self, records: List[NetworkRecord]):
        self.records = records
        self.map_center = records[0].antenna.position if records else (0, 0)
        self.unique_antennas = self._get_unique_antennas()

    def _get_unique_antennas(self) -> List[Tuple]:
        """Получение уникальных антенн на основе их позиций"""
        seen_positions = set()
        unique_antennas = []

        for record in self.records:
            pos = record.antenna.position
            pos_key = (round(pos[0], 6), round(pos[1], 6))  # Округление для сравнения

            if pos_key not in seen_positions:
                seen_positions.add(pos_key)
                unique_antennas.append(record.antenna)

        return unique_antennas

    def create_map(
        self,
        output_file: str = "network_coverage.html"
    ):
        """Создание интерактивной карты с антеннами и измерениями"""

        map_obj = folium.Map(location=self.map_center, zoom_start=13)

        # 1. Добавление кластеров для измерений
        lte1800_cluster = MarkerCluster(
            name="LTE1800 Measurements"
        ).add_to(map_obj)
        lte2100_cluster = MarkerCluster(
            name="LTE2100 Measurements"
        ).add_to(map_obj)

        # 2. Добавление измерений в соответствующие кластеры
        for record in self.records:
            popup_text = (
                f"Cell: {record.antenna.cellname}<br>"
                f"Band: {record.antenna.band}<br>"
                f"RSRP: {record.measurement.rsrp} dBm<br>"
                f"RSRQ: {record.measurement.rsrq} dB<br>"
                f"Time: {record.measurement.timestamp}"
            )

            marker = folium.CircleMarker(
                location=record.measurement.position,
                radius=3,
                color="green" if record.antenna.band == "LTE1800" else "purple",
                fill=True,
                popup=popup_text
            )

            if record.antenna.band == "LTE1800":
                marker.add_to(lte1800_cluster)
            else:
                marker.add_to(lte2100_cluster)

        # 3. Добавление антенн с расширенной информацией
        for antenna in self.unique_antennas:
            popup_text = (
                f"<b>{antenna.cellname}</b><br>"
                f"Band: {antenna.band}<br>"
                f"Azimuth: {antenna.azimuth}°<br>"
                f"Tilt: {antenna.tilt}°<br>"
                f"Height: {antenna.height}m<br>"
                f"Position: {antenna.position[0]:.6f}, {antenna.position[1]:.6f}"
            )

            folium.Marker(
                location=antenna.position,
                popup=popup_text,
                icon=folium.Icon(
                    color="blue" if antenna.band == "LTE1800" else "red",
                    icon="tower-cell",
                    prefix="fa"
                )
            ).add_to(map_obj)

        # 4. Тепловая карта RSRP
        heat_data = [
            [
                record.measurement.position[0],
                record.measurement.position[1],
                record.measurement.rsrp
            ]
            for record in self.records
        ]

        # Нормализация данных для лучшего отображения
        min_rsrp = min(record.measurement.rsrp for record in self.records)
        max_rsrp = max(record.measurement.rsrp for record in self.records)

        HeatMap(
            heat_data,
            name="RSRP Heatmap",
            min_opacity=0.3,
            radius=10,
            blur=5,
            gradient={
                0.0: "red",
                0.5: "yellow",
                1.0: "green"
            }
        ).add_to(map_obj)

        # Добавление слоя с зонами покрытия
        for antenna in self.unique_antennas:
            folium.Circle(
                location=antenna.position,
                radius=500,  # 500 метров
                color="blue" if antenna.band == "LTE1800" else "red",
                fill=True,
                fill_opacity=0.1,
                popup=f"{antenna.cellname} Coverage Zone"
            ).add_to(map_obj)

        # Управление слоями
        folium.LayerControl(collapsed=False).add_to(map_obj)

        # Добавление мини-карты для навигации
        folium.plugins.MiniMap(toggle_display=True).add_to(map_obj)

        # Сохранение карты
        map_obj.save(output_file)

        return map_obj
