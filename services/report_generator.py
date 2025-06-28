import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
from models.antenna import Antenna
from models.measurement import Measurement
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class ReportGenerator:
    def __init__(self, analysis_results: Dict, antenna_info: List[Antenna], 
                 coverage_threshold: float = -100, quality_threshold: float = -10):
        self.results = analysis_results
        self.antennas = antenna_info
        self.coverage_threshold = coverage_threshold
        self.quality_threshold = quality_threshold
    
    def generate_summary(self) -> str:
        """Генерация текстового отчета"""
        report = []
        
        # Общая информация
        report.append(f"Network Load Balancing Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of antennas: {len(self.antennas)}")
        report.append(f"Total measurements: {sum(self.results['measurement_counts'].values())}")
        report.append("\n--- Antenna Details ---")
        
        # Информация по антеннам
        for antenna in self.antennas:
            report.append(f"  - {antenna.cellname} ({antenna.band}): "
                          f"Azimuth={antenna.azimuth}°, Tilt={antenna.tilt}°, "
                          f"Location=({antenna.position[0]:.6f}, {antenna.position[1]:.6f})")
        
        # Результаты анализа
        report.append("\n--- Performance Summary ---")
        for band, metrics in self.results['band_metrics'].items():
            report.append(f"Band {band}:")
            report.append(f"  - Mean RSRP: {metrics['mean_rsrp']:.2f} dBm")
            report.append(f"  - Mean RSRQ: {metrics['mean_rsrq']:.2f} dB")
            report.append(f"  - Coverage (<{self.coverage_threshold} dBm): {metrics['coverage_probability']:.2%}")
            report.append(f"  - Quality (<{self.quality_threshold} dB): {metrics['quality_probability']:.2%}")
            report.append(f"  - Measurements: {self.results['measurement_counts'][band]}")
        
        # Проблемные зоны
        if self.results['weak_coverage_points']:
            report.append("\n--- Weak Coverage Areas ---")
            for i, point in enumerate(self.results['weak_coverage_points'][:5], 1):
                report.append(f"  {i}. Location: ({point[0]:.6f}, {point[1]:.6f}), "
                             f"RSRP: {point[2]} dBm, RSRQ: {point[3]} dB")
        
        return "\n".join(report)
    
    def generate_html_report(self, map_path: str, plot_paths: Dict[str, str]) -> str:
        """Генерация HTML отчета с визуализациями"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Network Load Balancing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .section {{ margin-bottom: 30px; }}
                .grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }}
                img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Network Load Balancing Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>Network Overview</h2>
                <div class="grid">
                    <div>
                        <h3>Antenna Locations</h3>
                        <img src="{map_path}" alt="Network Coverage Map">
                    </div>
                    <div>
                        <h3>Performance Summary</h3>
                        <table>
                            <tr>
                                <th>Band</th>
                                <th>Mean RSRP</th>
                                <th>Mean RSRQ</th>
                                <th>Coverage</th>
                                <th>Quality</th>
                            </tr>
        """
        
        # Таблица производительности
        for band, metrics in self.results['band_metrics'].items():
            html += f"""
            <tr>
                <td>{band}</td>
                <td>{metrics['mean_rsrp']:.2f} dBm</td>
                <td>{metrics['mean_rsrq']:.2f} dB</td>
                <td>{metrics['coverage_probability']:.2%}</td>
                <td>{metrics['quality_probability']:.2%}</td>
            </tr>
            """
        
        html += """
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Signal Distributions</h2>
                <div class="grid">
        """
        
        # Графики распределения
        for plot_name, plot_path in plot_paths.items():
            html += f"""
            <div>
                <h3>{plot_name.replace('_', ' ').title()}</h3>
                <img src="{plot_path}" alt="{plot_name}">
            </div>
            """
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Рекомендации
        recommendations = self.generate_recommendations()
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_recommendations(self) -> List[str]:
        """Генерация рекомендаций по балансировке нагрузки"""
        recs = []
        bands = list(self.results['band_metrics'].keys())
        
        if len(bands) < 2:
            return ["Insufficient data for load balancing recommendations"]
        
        # Сравнение производительности полос
        band1, band2 = bands[0], bands[1]
        metrics1 = self.results['band_metrics'][band1]
        metrics2 = self.results['band_metrics'][band2]
        
        # Проверка дисбаланса нагрузки
        load_diff = abs(metrics1['measurement_count'] - metrics2['measurement_count'])
        total_measurements = metrics1['measurement_count'] + metrics2['measurement_count']
        
        if load_diff / total_measurements > 0.3:
            recs.append(f"Significant load imbalance detected between {band1} and {band2} bands")
        
        # Рекомендации по покрытию
        if metrics1['coverage_probability'] < 0.8 or metrics2['coverage_probability'] < 0.8:
            recs.append("Optimize antenna tilt or azimuth to improve coverage in weak areas")
        
        # Рекомендации по качеству
        if metrics1['quality_probability'] < 0.7 or metrics2['quality_probability'] < 0.7:
            recs.append("Investigate interference sources and optimize network parameters")
        
        # Рекомендации по балансировке
        if metrics1['mean_rsrp'] > metrics2['mean_rsrp'] + 5:
            recs.append(f"Consider shifting some traffic from {band1} to {band2} for better load distribution")
        
        if not recs:
            recs.append("Network load is well balanced. Continue monitoring for changes")
        
        return recs