import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any, Optional, Tuple
from models.network_record import NetworkRecord


class SpatialVisualizer:
    """Класс для визуализации пространственных зависимостей сигнала"""

    @staticmethod
    def plot_spatial_dependencies(
        metrics: List[Dict[str, Any]],
        output_prefix: str = "spatial_dependencies",
        distance_bins: int = 20,
        azimuth_bins: int = 6,
        color_palette: str = "viridis",
        create_animation: bool = False,
        create_interactive: bool = False
    ) -> Dict[str, str]:
        """
        Визуализация пространственных зависимостей сигнала

        Параметры:
            metrics: Список словарей с метриками
            output_prefix: Префикс для имен файлов
            distance_bins: Количество бинов для гистограмм расстояния
            azimuth_bins: Количество бинов для разбиения азимута
            color_palette: Цветовая схема
            create_animation: Создавать ли анимацию для 3D-графика
            create_interactive: Создавать ли интерактивную 3D-визуализацию

        Возвращает:
            Словарь с путями к сохраненным файлам
        """
        if not metrics:
            print("Нет данных для визуализации пространственных зависимостей")
            return {}

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

        # Создаем папку для результатов
        os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

        # Словарь для путей к файлам
        output_files = {}

        # 1. RSRP vs Расстояние
        rsrp_dist_file = f"{output_prefix}_rsrp_vs_distance.png"
        SpatialVisualizer._plot_rsrp_vs_distance(
            df,
            rsrp_dist_file,
            color_palette
        )
        output_files['rsrp_vs_distance'] = rsrp_dist_file

        # 2. RSRQ vs Расстояние
        rsrq_dist_file = f"{output_prefix}_rsrq_vs_distance.png"
        SpatialVisualizer._plot_rsrq_vs_distance(
            df,
            rsrq_dist_file,
            color_palette
        )
        output_files['rsrq_vs_distance'] = rsrq_dist_file

        # 3. RSRP vs Разница азимута
        rsrp_az_file = f"{output_prefix}_rsrp_vs_azimuth.png"
        SpatialVisualizer._plot_rsrp_vs_azimuth_diff(
            df,
            rsrp_az_file,
            azimuth_bins,
            color_palette
        )
        output_files['rsrp_vs_azimuth'] = rsrp_az_file

        # 4. 3D-график RSRP vs Distance vs Azimuth
        rsrp_3d_file = f"{output_prefix}_3d_rsrp.png"
        SpatialVisualizer._plot_3d_rsrp(
            df,
            rsrp_3d_file,
            color_palette,
            create_animation
        )
        output_files['3d_rsrp'] = rsrp_3d_file

        # 5. Интерактивная 3D-визуализация
        if create_interactive:
            interactive_file = f"{output_prefix}_interactive_3d.html"
            SpatialVisualizer._plot_interactive_3d(df, interactive_file)
            output_files['interactive_3d'] = interactive_file

        # 6. Зависимость RSRP от направления (полярная диаграмма)
        polar_file = f"{output_prefix}_polar_rsrp.png"
        SpatialVisualizer._plot_polar_rsrp(df, polar_file)
        output_files['polar_rsrp'] = polar_file

        # 7. Числовые метрики с визуализацией
        metrics_files = SpatialVisualizer.visualize_rsrp_metrics(
            df,
            output_prefix
        )
        output_files.update(metrics_files)

        return output_files

    @staticmethod
    def _plot_rsrp_vs_distance(
        df: pd.DataFrame,
        output_file: str,
        color_palette: str = "viridis"
    ) -> None:
        """RSRP vs Расстояние: Точечный график с линиями тренда"""
        plt.figure(figsize=(12, 8))

        # Разбиение на зоны качества
        df['quality'] = pd.cut(df['rsrp'],
            bins=[-140, -110, -100, -90, -40],
            labels=['Очень плохо', 'Плохо', 'Удовлетворительно', 'Хорошо']
        )

        # Цветовая схема по качеству
        palette = {'Очень плохо': 'red', 'Плохо': 'orange',
                  'Удовлетворительно': 'yellow', 'Хорошо': 'green'}

        sns.scatterplot(
            data=df, x='distance', y='rsrp',
            hue='quality', palette=palette,
            alpha=0.7, s=50
        )

        # Линии тренда для каждого диапазона
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

        plt.title('Зависимость RSRP от расстояния до антенны')
        plt.xlabel('Расстояние (м)')
        plt.ylabel('RSRP (dBm)')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Качество сигнала')
        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Сохранен график RSRP vs Расстояние: {output_file}")

    @staticmethod
    def _plot_rsrq_vs_distance(
        df: pd.DataFrame,
        output_file: str,
        color_palette: str = "viridis"
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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

    @staticmethod
    def _plot_polar_rsrp(
        df: pd.DataFrame,
        output_file: str = "polar_rsrp.png"
    ) -> None:
        """Полярная диаграмма зависимости RSRP от направления"""
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111, projection='polar')

        # Нормализация RSRP для визуализации
        df['norm_rsrp'] = (df['rsrp'] - df['rsrp'].min()) / (df['rsrp'].max() - df['rsrp'].min())

        # Размер точки в зависимости от RSRP
        sizes = 50 + 150 * df['norm_rsrp']

        # Цвета по диапазонам
        colors = df['band'].map({'LTE1800': 'blue', 'LTE2100': 'red'})

        # Преобразование азимута в радианы
        theta = np.radians(df['azimuth_diff'])

        # Визуализация
        scatter = ax.scatter(
            theta, df['distance'],
            c=colors, s=sizes,
            alpha=0.7, cmap='viridis'
        )

        # Аннотации
        ax.set_title('RSRP в зависимости от направления и расстояния', pad=20)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(45)

        # Легенда
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='LTE1800'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='LTE2100')
        ]
        ax.legend(handles=legend_elements, loc='lower right', title='Частотный диапазон')

        # Цветовая шкала для RSRP
        norm = plt.Normalize(df['rsrp'].min(), df['rsrp'].max())
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('RSRP (dBm)')

        plt.savefig(output_file, dpi=150)
        plt.close()
        print(f"Сохранена полярная диаграмма: {output_file}")

    @staticmethod
    def visualize_rsrp_metrics(
        df: pd.DataFrame,
        output_prefix: str = "rsrp_metrics"
    ) -> Dict[str, str]:
        """Визуализация числовых метрик RSRP"""
        output_files = {}

        # 1. Сила сигнала
        strength_file = f"{output_prefix}_signal_strength.png"
        strength_data = SpatialVisualizer._visualize_signal_strength(df, strength_file)
        output_files['signal_strength'] = strength_file

        # 2. Стабильность покрытия
        stability_file = f"{output_prefix}_coverage_stability.png"
        stability_data = SpatialVisualizer._visualize_coverage_stability(df, stability_file)
        output_files['coverage_stability'] = stability_file

        # 3. Проблемные зоны
        problems_file = f"{output_prefix}_problem_areas.png"
        problems_data = SpatialVisualizer._visualize_problem_areas(df, problems_file)
        output_files['problem_areas'] = problems_file

        # 4. Сравнение диапазонов
        comparison_file = f"{output_prefix}_band_comparison.png"
        comparison_data = SpatialVisualizer._visualize_band_comparison(df, comparison_file)
        output_files['band_comparison'] = comparison_file

        # 5. Сохранение числовых данных
        metrics_df = pd.DataFrame({
            'band': strength_data['band'],
            'median_rsrp': strength_data['median'],
            'iqr': stability_data['IQR'],
            'problem_percent': problems_data['percent'],
            'problem_count': problems_data['count'],
            'stability_ratio': comparison_data['stability_ratio']
        })
        csv_file = f"{output_prefix}_metrics.csv"
        metrics_df.to_csv(csv_file, index=False)
        output_files['metrics_csv'] = csv_file

        return output_files

    @staticmethod
    def _visualize_signal_strength(
        df: pd.DataFrame,
        output_file: str
    ) -> pd.DataFrame:
        """Визуализация силы сигнала"""
        # Расчет метрик
        metrics = df.groupby('band')['rsrp'].agg(['median', 'mean', 'std']).reset_index()
        metrics['diff_median'] = metrics['median'].diff().fillna(0)

        # Визуализация
        plt.figure(figsize=(10, 6))

        # Столбчатая диаграмма медиан
        bars = plt.bar(metrics['band'], metrics['median'], color=['blue', 'red'])

        # Добавление значений
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.1f} dBm',
                     ha='center', va='bottom')

        plt.title('Медианные значения RSRP по диапазонам')
        plt.xlabel('Частотный диапазон')
        plt.ylabel('RSRP (dBm)')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(output_file, dpi=150)
        plt.close()

        return metrics

    @staticmethod
    def _visualize_coverage_stability(
        df: pd.DataFrame,
        output_file: str
    ) -> pd.DataFrame:
        """Визуализация стабильности покрытия"""
        # Расчет метрик
        def iqr(x):
            return x.quantile(0.75) - x.quantile(0.25)

        stability = df.groupby('band')['rsrp'].agg([iqr, lambda x: x.quantile(0.75)/x.quantile(0.25)]).reset_index()
        stability.columns = ['band', 'IQR', 'Q3/Q1']

        # Визуализация
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # IQR
        ax1.bar(stability['band'], stability['IQR'], color=['blue', 'red'], alpha=0.7)
        ax1.set_ylabel('Межквартильный размах (IQR)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        # Q3/Q1
        ax2 = ax1.twinx()
        ax2.plot(stability['band'], stability['Q3/Q1'], 'o-', color='green', linewidth=2)
        ax2.set_ylabel('Отношение Q3/Q1', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        plt.title('Стабильность покрытия по диапазонам')
        plt.savefig(output_file, dpi=150)
        plt.close()

        return stability

    @staticmethod
    def _visualize_problem_areas(
        df: pd.DataFrame,
        output_file: str
    ) -> pd.DataFrame:
        """Визуализация проблемных зон"""
        # Расчет метрик
        df['problem'] = df['rsrp'] < -100
        problem_metrics = df.groupby('band')['problem'].agg(['mean', 'sum']).reset_index()
        problem_metrics['mean'] *= 100  # Проценты

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Процент проблемных измерений
        ax1.bar(problem_metrics['band'], problem_metrics['mean'], color=['blue', 'red'])
        ax1.set_title('Процент измерений < -100 dBm')
        ax1.set_ylabel('Процент (%)')
        ax1.set_ylim(0, 100)

        # Добавление значений на столбцы
        for i, v in enumerate(problem_metrics['mean']):
            ax1.text(i, v + 1, f"{v:.1f}%", ha='center')

        # Количество выбросов
        ax2.bar(problem_metrics['band'], problem_metrics['sum'], color=['blue', 'red'])
        ax2.set_title('Количество проблемных измерений')
        ax2.set_ylabel('Количество')

        # Добавление значений на столбцы
        for i, v in enumerate(problem_metrics['sum']):
            ax2.text(i, v + 5, f"{v}", ha='center')

        plt.suptitle('Анализ проблемных зон покрытия')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        return problem_metrics.rename(columns={'mean': 'percent', 'sum': 'count'})

    @staticmethod
    def _visualize_band_comparison(
        df: pd.DataFrame,
        output_file: str
    ) -> pd.DataFrame:
        """Визуализация сравнения диапазонов"""
        # Расчет метрик
        metrics = df.groupby('band')['rsrp'].agg(['median', lambda x: x.quantile(0.75) - x.quantile(0.25)])
        metrics.columns = ['median', 'IQR']
        metrics['stability_ratio'] = metrics['IQR'].min() / metrics['IQR']  # Относительная стабильность

        # Визуализация
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Разница медиан
        median_diff = metrics['median'].iloc[0] - metrics['median'].iloc[1]
        ax1.bar(['Разница медиан'], [abs(median_diff)], color='purple')
        ax1.set_title(f'Разница медиан: {abs(median_diff):.1f} dBm')
        ax1.set_ylabel('ΔRSRP (dBm)')
        ax1.text(0, abs(median_diff)/2, f"{abs(median_diff):.1f} dBm", ha='center', color='white', fontsize=12)

        # Относительная стабильность
        ax2.bar(metrics.index, metrics['stability_ratio'], color=['blue', 'red'])
        ax2.set_title('Относительная стабильность покрытия')
        ax2.set_ylabel('Коэффициент стабильности')
        ax2.axhline(1, color='gray', linestyle='--')

        # Добавление значений на столбцы
        for i, v in enumerate(metrics['stability_ratio']):
            ax2.text(i, v + 0.05, f"{v:.2f}", ha='center')

        plt.suptitle('Сравнение диапазонов')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        return metrics
