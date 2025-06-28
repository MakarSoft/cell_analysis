import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Tuple, Dict
from models.network_record import NetworkRecord
import pandas as pd

class HeatmapGenerator:
    def __init__(self, grid_size: float = 0.0005, rsrp_range: Tuple[float, float] = (-120, -70)):
        self.grid_size = grid_size  # Размер ячейки сетки в градусах
        self.rsrp_range = rsrp_range
    
    def create_heatmap_data(self, records: List[NetworkRecord]) -> Dict[str, np.ndarray]:
        """Подготовка данных для тепловой карты"""
        # Создание DataFrame для удобства обработки
        data = {
            'latitude': [r.measurement.position[0] for r in records],
            'longitude': [r.measurement.position[1] for r in records],
            'rsrp': [r.measurement.rsrp for r in records],
            'band': [r.antenna.band for r in records]
        }
        df = pd.DataFrame(data)
        
        # Создание сетки
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # Добавление буфера вокруг данных
        lat_min -= self.grid_size * 2
        lat_max += self.grid_size * 2
        lon_min -= self.grid_size * 2
        lon_max += self.grid_size * 2
        
        # Создание координатной сетки
        lat_bins = np.arange(lat_min, lat_max, self.grid_size)
        lon_bins = np.arange(lon_min, lon_max, self.grid_size)
        
        # Инициализация матриц
        heatmap_data = {
            'all': np.zeros((len(lat_bins), len(lon_bins))),
            'LTE1800': np.zeros((len(lat_bins), len(lon_bins))),
            'LTE2100': np.zeros((len(lat_bins), len(lon_bins)))
        }
        count_maps = {k: np.zeros_like(v) for k, v in heatmap_data.items()}
        
        # Заполнение матриц
        for _, row in df.iterrows():
            lat_idx = np.digitize(row['latitude'], lat_bins) - 1
            lon_idx = np.digitize(row['longitude'], lon_bins) - 1
            
            if 0 <= lat_idx < len(lat_bins) and 0 <= lon_idx < len(lon_bins):
                heatmap_data['all'][lat_idx, lon_idx] += row['rsrp']
                heatmap_data[row['band']][lat_idx, lon_idx] += row['rsrp']
                
                count_maps['all'][lat_idx, lon_idx] += 1
                count_maps[row['band']][lat_idx, lon_idx] += 1
        
        # Расчет средних значений
        for key in heatmap_data:
            mask = count_maps[key] > 0
            heatmap_data[key][mask] /= count_maps[key][mask]
            heatmap_data[key][~mask] = np.nan
        
        return {
            'data': heatmap_data,
            'lat_bins': lat_bins,
            'lon_bins': lon_bins,
            'antennas': [(a.position[0], a.position[1]) for a in set(r.antenna for r in records)]
        }
    
    def plot_heatmap(self, heatmap_data: Dict, band: str = 'all', 
                     output_file: str = 'heatmap.png', title: str = 'RSRP Heatmap'):
        """Визуализация тепловой карты"""
        plt.figure(figsize=(12, 10))
        
        # Кастомная цветовая карта (от красного к зеленому)
        cmap = LinearSegmentedColormap.from_list('rsrp_cmap', ['red', 'yellow', 'green'])
        
        # Данные для выбранной полосы
        data = heatmap_data['data'][band]
        lat_bins = heatmap_data['lat_bins']
        lon_bins = heatmap_data['lon_bins']
        
        # Визуализация
        plt.pcolormesh(lon_bins, lat_bins, data, 
                       cmap=cmap, vmin=self.rsrp_range[0], vmax=self.rsrp_range[1],
                       shading='auto')
        
        # Антенны
        for lat, lon in heatmap_data['antennas']:
            plt.scatter(lon, lat, c='blue', s=100, marker='^', edgecolors='white')
        
        # Настройки
        plt.colorbar(label='RSRP (dBm)')
        plt.title(f'{title} - {band}' if band != 'all' else title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(alpha=0.3)
        
        # Сохранение
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_file
    
    def generate_all_heatmaps(self, records: List[NetworkRecord], output_dir: str = 'heatmaps'):
        """Генерация всех тепловых карт"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Подготовка данных
        heatmap_data = self.create_heatmap_data(records)
        
        # Генерация карт
        files = {}
        for band in ['all', 'LTE1800', 'LTE2100']:
            if band == 'all' or band in heatmap_data['data']:
                output_file = os.path.join(output_dir, f'{band.lower()}_heatmap.png')
                title = 'RSRP Heatmap' if band == 'all' else f'{band} RSRP Heatmap'
                files[band] = self.plot_heatmap(heatmap_data, band, output_file, title)
        
        return files