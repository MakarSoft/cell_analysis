import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional
from models.network_record import NetworkRecord
from matplotlib.animation import FuncAnimation


class PlotVisualizer:


    @staticmethod
    def plot_signal_distributions_separate(
        records: List[NetworkRecord],
        output_prefix: str = "signal_distributions"
    ):
        """
        Сохранение каждого графика распределения сигнала в отдельный файл.

        Параметры:
            records: Список объектов NetworkRecord
            output_prefix: Префикс для имен файлов
        """
        # Создаем DataFrame из записей
        data = {
            'band': [r.antenna.band for r in records],
            'rsrp': [r.measurement.rsrp for r in records],
            'rsrq': [r.measurement.rsrq for r in records]
        }
        df = pd.DataFrame(data)

        # Создаем папку для результатов
        os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

        # 1. Гистограмма RSRP
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='rsrp', hue='band', kde=True, element='step')
        plt.title('Распределение RSRP')
        plt.xlabel('RSRP (dBm)')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrp_hist.png", dpi=150)
        plt.close()
        print(f"Сохранена гистограмма RSRP: {output_prefix}_rsrp_hist.png")

        # 2. Boxplot RSRP
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='band', y='rsrp')
        plt.title('Распределение RSRP по диапазонам')
        plt.xlabel('Диапазон')
        plt.ylabel('RSRP (dBm)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrp_boxplot.png", dpi=150)
        plt.close()
        print(f"Сохранен боксплот RSRP: {output_prefix}_rsrp_boxplot.png")

        # 3. Гистограмма RSRQ
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='rsrq', hue='band', kde=True, element='step')
        plt.title('Распределение RSRQ')
        plt.xlabel('RSRQ (dB)')
        plt.ylabel('Частота')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrq_hist.png", dpi=150)
        plt.close()
        print(f"Сохранена гистограмма RSRQ: {output_prefix}_rsrq_hist.png")

        # 4. Boxplot RSRQ
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='band', y='rsrq')
        plt.title('Распределение RSRQ по диапазонам')
        plt.xlabel('Диапазон')
        plt.ylabel('RSRQ (dB)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrq_boxplot.png", dpi=150)
        plt.close()
        print(f"Сохранен боксплот RSRQ: {output_prefix}_rsrq_boxplot.png")

        # 5. Совмещенный график плотности RSRP
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='rsrp', hue='band', fill=True, alpha=0.3)
        plt.title('Плотность распределения RSRP')
        plt.xlabel('RSRP (dBm)')
        plt.ylabel('Плотность')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrp_density.png", dpi=150)
        plt.close()
        print(f"Сохранена плотность RSRP: {output_prefix}_rsrp_density.png")

        # 6. Совмещенный график плотности RSRQ
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='rsrq', hue='band', fill=True, alpha=0.3)
        plt.title('Плотность распределения RSRQ')
        plt.xlabel('RSRQ (dB)')
        plt.ylabel('Плотность')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_rsrq_density.png", dpi=150)
        plt.close()
        print(f"Сохранена плотность RSRQ: {output_prefix}_rsrq_density.png")

    @staticmethod
    def plot_signal_distributions(
        records: List[NetworkRecord],
        output_file: str = "signal_distributions.png"
    ):
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        # Разделение данных по частотам
        lte1800 = [r for r in records if r.antenna.band == "LTE1800"]
        lte2100 = [r for r in records if r.antenna.band == "LTE2100"]

        # Гистограммы RSRP
        sns.histplot(
            [r.measurement.rsrp for r in lte1800],
            label="LTE1800", kde=True, ax=ax[0, 0]
        )
        sns.histplot(
            [r.measurement.rsrp for r in lte2100],
            label="LTE2100", kde=True, ax=ax[0, 0]
        )
        ax[0, 0].set_title("RSRP Distribution")
        ax[0, 0].legend()

        # Boxplot RSRP
        sns.boxplot(
            x=[r.antenna.band for r in records],
            y=[r.measurement.rsrp for r in records],
            ax=ax[0, 1]
        )
        ax[0, 1].set_title("RSRP by Band")

        # Аналогично для RSRQ...
        # [код для RSRQ графиков]

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()

    @staticmethod
    def plot_spatial_dependencies(
        metrics: List[Dict[str, Any]],
        output_prefix: str = "spatial_dependencies",
        distance_bins: int = 20,
        azimuth_bins: int = 6,
        color_palette: str = "viridis",
        create_animation: bool = False,
        create_interactive: bool = False
    ):
        """
        Визуализация пространственных зависимостей сигнала с сохранением
        каждого графика отдельно

        Параметры:
            metrics: Список словарей с метриками
            output_prefix: Префикс для имен файлов
            distance_bins: Количество бинов для гистограмм расстояния
            azimuth_bins: Количество бинов для разбиения азимута
            color_palette: Цветовая схема
            create_animation: Создавать ли анимацию для 3D-графика
            create_interactive: Создавать ли интерактивную 3D-визуализацию
        """
        if not metrics:
            print("Нет данных для визуализации пространственных зависимостей")
            return

        # Подготовка данных
        data = []
        for m in metrics:
            record = m['record']
            data.append({
                'distance': m['distance'],
                'azimuth_diff': m['azimuth_diff'],
                'rsrp': record.measurement.rsrp,
                'rsrq': record.measurement.rsrq,
                'band': record.antenna.band,
                'antenna': record.antenna.cellname
            })

        df = pd.DataFrame(data)

        # Создаем папку для результатов, если нужно
        os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

        # 1. RSRP vs Расстояние
        PlotVisualizer._plot_rsrp_vs_distance(
            df,
            f"{output_prefix}_rsrp_vs_distance.png",
            color_palette
        )

        # 2. RSRQ vs Расстояние
        PlotVisualizer._plot_rsrq_vs_distance(
            df,
            f"{output_prefix}_rsrq_vs_distance.png",
            color_palette
        )

        # 3. RSRP vs Разница азимута
        PlotVisualizer._plot_rsrp_vs_azimuth_diff(
            df,
            f"{output_prefix}_rsrp_vs_azimuth.png",
            azimuth_bins,
            color_palette
        )

        # 4. 3D-график RSRP vs Distance vs Azimuth
        PlotVisualizer._plot_3d_rsrp(
            df,
            f"{output_prefix}_3d_rsrp.png",
            color_palette,
            create_animation
        )

        # 5. Интерактивная 3D-визуализация
        if create_interactive:
            PlotVisualizer._plot_interactive_3d(
                df,
                f"{output_prefix}_interactive_3d.html"
            )

    @staticmethod
    def _plot_rsrp_vs_distance(
        df: pd.DataFrame,
        output_file: str,
        color_palette: str = "viridis"
    ):
        """RSRP vs Расстояние: Точечный график с линиями тренда"""
        plt.figure(figsize=(10, 6))

        # Точечный график
        sns.scatterplot(
            data=df, x='distance', y='rsrp', hue='band',
            alpha=0.6, palette=color_palette
        )

        # Линии тренда
        for band in df['band'].unique():
            band_data = df[df['band'] == band]
            if len(band_data) > 1:
                z = np.polyfit(band_data['distance'], band_data['rsrp'], 1)
                p = np.poly1d(z)
                plt.plot(
                    band_data['distance'], p(band_data['distance']),
                    linewidth=2, linestyle='--',
                    label=f'{band} тренд'
                )

        plt.title('RSRP vs Расстояние до антенны')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('RSRP (dBm)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Сохранен график RSRP vs Расстояние: {output_file}")

    @staticmethod
    def _plot_rsrq_vs_distance(
        df: pd.DataFrame,
        output_file: str,
        color_palette: str = "viridis"
    ):
        """RSRQ vs Расстояние: Точечный график"""
        plt.figure(figsize=(10, 6))

        sns.scatterplot(
            data=df, x='distance', y='rsrq', hue='band',
            alpha=0.6, palette=color_palette
        )

        plt.title('RSRQ vs Расстояние до антенны')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('RSRQ (dB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Сохранен график RSRQ vs Расстояние: {output_file}")

    @staticmethod
    def _plot_rsrp_vs_azimuth_diff(
        df: pd.DataFrame,
        output_file: str,
        azimuth_bins: int = 6,
        color_palette: str = "viridis"
    ):
        """RSRP vs Разница азимута: Боксплоты"""
        plt.figure(figsize=(12, 8))

        # Разбиваем на бины
        df['azimuth_bin'] = pd.cut(
            df['azimuth_diff'],
            bins=azimuth_bins,
            labels=[f"Бин {i+1}" for i in range(azimuth_bins)]
        )

        sns.boxplot(
            data=df, x='band', y='rsrp', hue='azimuth_bin',
            palette=color_palette, showfliers=False
        )

        plt.title('RSRP vs Разница азимута')
        plt.xlabel('Частотный диапазон')
        plt.ylabel('RSRP (dBm)')
        plt.legend(title='Разница азимута (°)', loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Сохранен график RSRP vs Азимут: {output_file}")

    @staticmethod
    def _plot_3d_rsrp(
        df: pd.DataFrame,
        output_file: str,
        color_palette: str = "viridis",
        create_animation: bool = False
    ):
        """3D-график: RSRP vs Расстояние и Азимут"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Цвета по диапазонам
        colors = df['band'].map({'LTE1800': 'blue', 'LTE2100': 'red'})

        # 3D-точечный график
        scatter = ax.scatter(
            df['distance'],
            df['azimuth_diff'],
            df['rsrp'],
            c=colors,
            alpha=0.6,
            depthshade=True
        )

        ax.set_title('RSRP vs Расстояние и Азимут')
        ax.set_xlabel('Расстояние (м)')
        ax.set_ylabel('Разница азимута (°)')
        ax.set_zlabel('RSRP (dBm)')

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='LTE1800'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='LTE2100')
        ]
        ax.legend(handles=legend_elements)

        # Начальный угол обзора
        ax.view_init(elev=30, azim=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Сохранен 3D-график: {output_file}")

        # Анимация вращения
        if create_animation:
            animation_dir = os.path.join(os.path.dirname(output_file), "3d_animation")
            os.makedirs(animation_dir, exist_ok=True)

            def update(frame):
                ax.view_init(elev=30, azim=frame)
                return [scatter]

            anim = FuncAnimation(
                fig, update, frames=np.arange(0, 360, 5),
                interval=50, blit=True
            )

            # Сохранение анимации
            anim_path = os.path.join(animation_dir, "3d_rotation.gif")
            anim.save(anim_path, writer='pillow', fps=15, dpi=100)
            print(f"Сохранена анимация: {anim_path}")

        plt.close()

    @staticmethod
    def _plot_interactive_3d(
        df: pd.DataFrame,
        output_file: str = "interactive_spatial.html"
    ):
        """Интерактивная 3D-визуализация с использованием Plotly"""
        try:
            import plotly.express as px
        except ImportError:
            print("Для интерактивной визуализации установите plotly: pip install plotly")
            return

        fig = px.scatter_3d(
            df,
            x='distance',
            y='azimuth_diff',
            z='rsrp',
            color='band',
            symbol='antenna',
            hover_name='rsrq',
            hover_data={
                'distance': ':.1f',
                'azimuth_diff': ':.1f',
                'rsrp': ':.1f',
                'rsrq': ':.1f',
                'band': True,
                'antenna': True
            },
            opacity=0.7,
            title='RSRP vs Расстояние и Азимут',
            labels={
                'distance': 'Расстояние (м)',
                'azimuth_diff': 'Разница азимута (°)',
                'rsrp': 'RSRP (dBm)'
            },
            color_discrete_map={
                'LTE1800': 'blue',
                'LTE2100': 'red'
            }
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='Расстояние (м)',
                yaxis_title='Разница азимута (°)',
                zaxis_title='RSRP (dBm)'
            ),
            legend_title_text='Диапазон и Антенна',
            margin=dict(l=0, r=0, b=0, t=30)
        )

        fig.write_html(output_file)
        print(f"Сохранена интерактивная 3D-визуализация: {output_file}")
