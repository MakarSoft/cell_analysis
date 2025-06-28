from services.data_loader import DataLoader
from services.spatial_calculator import SpatialCalculator
from visualization.map_visualizer import MapVisualizer
from visualization.plot_visualizer import PlotVisualizer
from visualization.spatial_visualizer import SpatialVisualizer
from analysis.signal_analyzer import SignalAnalyzer


def main():
    # Загрузка данных
    loader = DataLoader("network_measurements.csv")
    records = loader.load_data()

    print(f"Loaded {len(records)} valid records")

    # Визуализация на карте
    map_vis = MapVisualizer(records)
    map_vis.create_map("network_coverage.html")

    # Анализ сигналов
    band_performance = SignalAnalyzer.analyze_band_performance(records)
    for band, metrics in band_performance.items():
        print(f"\n{band} Performance:")
        print(f"  Mean RSRP: {metrics['mean_rsrp']:.2f} dBm")
        print(f"  Coverage Probability: {metrics['coverage_probability']:.2%}")

    # Пространственный анализ
    spatial_metrics = SpatialCalculator.calculate_metrics(records)

    # Визуализация распределений
    PlotVisualizer.plot_signal_distributions(records)
    # Анализ распределений сигналов (отдельные файлы)
    PlotVisualizer.plot_signal_distributions_separate(
        records,
        output_prefix="results/signal_distributions"
    )
    # PlotVisualizer.plot_spatial_dependencies(
    #     spatial_metrics,
    #     create_animation=True,
    #     create_interactive=True
    # )

    PlotVisualizer.plot_spatial_dependencies(spatial_metrics)

    # ---------------------------------------------------

    # Расчет пространственных метрик
    spatial_metrics = SpatialCalculator.calculate_metrics(records)

    # Визуализация зависимостей
    visualizations = SpatialVisualizer.plot_spatial_dependencies(
        spatial_metrics,
        output_prefix="results/spatial/analysis",
        distance_bins=15,
        azimuth_bins=8,
        create_animation=True,
        create_interactive=True
    )

    print("Созданные визуализации:")
    for name, path in visualizations.items():
        print(f"- {name}: {path}")



if __name__ == "__main__":
    main()
