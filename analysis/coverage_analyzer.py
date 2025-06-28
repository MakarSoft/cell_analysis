import numpy as np
from typing import List, Dict, Tuple
from models.network_record import NetworkRecord
from scipy.spatial import KDTree

class CoverageAnalyzer:
    def __init__(self, coverage_threshold: float = -100, quality_threshold: float = -10):
        self.coverage_threshold = coverage_threshold
        self.quality_threshold = quality_threshold
    
    def analyze_coverage(self, records: List[NetworkRecord]) -> Dict:
        """Анализ покрытия сети"""
        results = {
            'band_metrics': {},
            'measurement_counts': {},
            'weak_coverage_points': [],
            'weak_quality_points': []
        }
        
        # Группировка по частотам
        band_data = {}
        for record in records:
            band = record.antenna.band
            if band not in band_data:
                band_data[band] = []
            band_data[band].append(record)
        
        # Расчет метрик для каждой полосы
        for band, band_records in band_data.items():
            rsrp_values = [r.measurement.rsrp for r in band_records]
            rsrq_values = [r.measurement.rsrq for r in band_records]
            
            coverage_prob = np.mean(np.array(rsrp_values) > self.coverage_threshold)
            quality_prob = np.mean(np.array(rsrq_values) > self.quality_threshold)
            
            results['band_metrics'][band] = {
                'mean_rsrp': np.mean(rsrp_values),
                'std_rsrp': np.std(rsrp_values),
                'mean_rsrq': np.mean(rsrq_values),
                'coverage_probability': coverage_prob,
                'quality_probability': quality_prob,
                'measurement_count': len(band_records)
            }
            
            results['measurement_counts'][band] = len(band_records)
            
            # Сбор точек со слабым покрытием
            for record in band_records:
                if record.measurement.rsrp < self.coverage_threshold:
                    results['weak_coverage_points'].append((
                        record.measurement.position[0],
                        record.measurement.position[1],
                        record.measurement.rsrp,
                        record.measurement.rsrq,
                        band
                    ))
                if record.measurement.rsrq < self.quality_threshold:
                    results['weak_quality_points'].append((
                        record.measurement.position[0],
                        record.measurement.position[1],
                        record.measurement.rsrp,
                        record.measurement.rsrq,
                        band
                    ))
        
        # Кластеризация слабых зон покрытия
        results['coverage_clusters'] = self.cluster_weak_points(results['weak_coverage_points'])
        results['quality_clusters'] = self.cluster_weak_points(results['weak_quality_points'])
        
        return results
    
    def cluster_weak_points(self, points: List[Tuple], radius: float = 0.001) -> List[Dict]:
        """Кластеризация точек со слабым покрытием"""
        if not points:
            return []
        
        # Создание KD-дерева для пространственного поиска
        coords = np.array([(point[0], point[1]) for point in points])
        kdtree = KDTree(coords)
        
        clusters = []
        visited = set()
        
        for i, point in enumerate(coords):
            if i in visited:
                continue
                
            # Поиск соседей в радиусе
            neighbors = kdtree.query_ball_point(point, radius)
            if not neighbors:
                continue
                
            # Создание кластера
            cluster_points = [points[idx] for idx in neighbors]
            avg_lat = np.mean([p[0] for p in cluster_points])
            avg_lon = np.mean([p[1] for p in cluster_points])
            min_rsrp = min([p[2] for p in cluster_points])
            min_rsrq = min([p[3] for p in cluster_points])
            bands = set([p[4] for p in cluster_points])
            
            clusters.append({
                'center': (avg_lat, avg_lon),
                'radius': radius,
                'point_count': len(cluster_points),
                'min_rsrp': min_rsrp,
                'min_rsrq': min_rsrq,
                'affected_bands': list(bands)
            })
            
            # Пометить точки как посещенные
            visited.update(neighbors)
        
        # Сортировка кластеров по размеру
        clusters.sort(key=lambda x: x['point_count'], reverse=True)
        return clusters
    
    def calculate_coverage_overlap(self, records: List[NetworkRecord]) -> Dict:
        """Анализ перекрытия зон покрытия разных частот"""
        band_points = {}
        
        # Группировка точек по частотам
        for record in records:
            band = record.antenna.band
            if band not in band_points:
                band_points[band] = []
            band_points[band].append(record.measurement.position)
        
        # Расчет перекрытия
        overlap_results = {}
        bands = list(band_points.keys())
        
        for i, band1 in enumerate(bands):
            for band2 in bands[i+1:]:
                key = f"{band1}-{band2}"
                
                # Создание KD-деревьев для быстрого поиска соседей
                tree1 = KDTree(band_points[band1])
                tree2 = KDTree(band_points[band2])
                
                # Поиск точек в радиусе 50 метров (примерно 0.00045 градуса)
                overlap_count = 0
                for point in band_points[band1]:
                    neighbors = tree2.query_ball_point(point, 0.00045)
                    if neighbors:
                        overlap_count += 1
                
                overlap_percent = overlap_count / len(band_points[band1])
                overlap_results[key] = {
                    'overlap_percent': overlap_percent,
                    'band1_count': len(band_points[band1]),
                    'band2_count': len(band_points[band2]),
                    'overlap_count': overlap_count
                }
        
        return overlap_results