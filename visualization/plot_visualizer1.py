import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, List, Dict
from models.network_record import NetworkRecord


class PlotVisualizer:

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
        output_file: str = "spatial_dependencies.png"
    ):
        """
        Визуализация пространственных зависимостей сигнала
        (реализация графиков зависимости сигнала от расстояния и угла)

        Параметры:
            metrics: Список словарей с ключами:
                - 'record': NetworkRecord
                - 'distance': расстояние до антенны (м)
                - 'azimuth_diff': разница азимута (градусы)
            output_file: Путь для сохранения графика
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

        df = pd.DataFrame(data)     # Создание DataFrame

        # Создание фигуры с 4 подграфиками
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Пространственные зависимости сигнала', fontsize=16)

        # 1. RSRP vs Distance
        sns.scatterplot(
            data=df, x='distance', y='rsrp', hue='band',
            alpha=0.6, palette="viridis", ax=axes[0, 0]
        )
        axes[0, 0].set_title('RSRP vs Расстояние до антенны')
        axes[0, 0].set_xlabel('Расстояние (м)')
        axes[0, 0].set_ylabel('RSRP (dBm)')
        axes[0, 0].grid(True, alpha=0.3)

        # Линия тренда
        for band in df['band'].unique():
            band_data = df[df['band'] == band]
            if len(band_data) > 1:  # Проверка наличия достаточных данных
                z = np.polyfit(band_data['distance'], band_data['rsrp'], 1)
                p = np.poly1d(z)
                axes[0, 0].plot(
                    band_data['distance'], p(band_data['distance']),
                    linewidth=2, linestyle='--',
                    label=f'{band} тренд'
                )

        # 2. RSRQ vs Distance
        sns.scatterplot(
            data=df, x='distance', y='rsrq', hue='band',
            alpha=0.6, palette="viridis", ax=axes[0, 1]
        )
        axes[0, 1].set_title('RSRQ vs Расстояние до антенны')
        axes[0, 1].set_xlabel('Расстояние (м)')
        axes[0, 1].set_ylabel('RSRQ (dB)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. RSRP vs Azimuth Difference
        sns.boxplot(
            data=df, x='band', y='rsrp', hue=pd.cut(df['azimuth_diff'], bins=6),
            palette="coolwarm", ax=axes[1, 0], showfliers=False
        )
        axes[1, 0].set_title('RSRP vs Разница азимута')
        axes[1, 0].set_xlabel('Частотный диапазон')
        axes[1, 0].set_ylabel('RSRP (dBm)')
        axes[1, 0].legend(title='Разница азимута (°)', loc='upper right')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 3D-график RSRP vs Distance vs Azimuth
        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')

        # Цвета по диапазонам
        colors = df['band'].map({'LTE1800': 'blue', 'LTE2100': 'red'})

        # 3D-точечный график
        scatter = ax_3d.scatter(
            df['distance'],
            df['azimuth_diff'],
            df['rsrp'],
            c=colors,
            alpha=0.6,
            depthshade=True
        )

        ax_3d.set_title('RSRP vs Расстояние и Азимут')
        ax_3d.set_xlabel('Расстояние (м)')
        ax_3d.set_ylabel('Разница азимута (°)')
        ax_3d.set_zlabel('RSRP (dBm)')

        # Легенда для цветов
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='LTE1800'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='LTE2100')
        ]
        ax_3d.legend(handles=legend_elements)

        # Оптимизация расположения
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Сохранение и закрытие
        plt.savefig(output_file, dpi=150)
        plt.close()

        return output_file
