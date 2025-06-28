## Проект анализа данных балансировки нагрузки на узел сотовой сети.
Включает загрузку данных в определенном формате, визуализацию на карте,
анализ распределений и зависимостей.

- Загрузка и предобработка данных (Pandas)
- Визуализация на карте (Folium, Plotly, KeplerGL)
- Графики распределения RSRP/RSRQ (Matplotlib, Seaborn)
- Анализ зависимости RSRP от частоты и параметров антенн
- Оптимизация tilt/azimuth для балансировки нагрузки

```
load_balancing_analysis/
├── models/                     - Модели данных
│   ├── __init__.py
│   ├── antenna.py
│   ├── measurement.py
│   └── network_record.py
├── services/                   - Сервисы
│   ├── __init__.py
│   ├── data_loader.py
│   ├── spatial_calculator.py
│   └── report_generator.py         Генератор отчетов
├── visualization/              - Визуализация
│   ├── __init__.py
│   ├── map_visualizer.py
│   ├── plot_visualizer.py
│   └── heatmap_generator.py        Генератор тепловых карт
├── analysis/                   - Анализ
│   ├── __init__.py
│   ├── signal_analyzer.py
│   └── coverage_analyzer.py        Анализатор покрытия
├── main.py                     Главный модуль
├── config.py
└── requirements.txt
```

---

## Запуск:
- Создание и активация виртуального окружения, установка зависимостей и запуск...
```bash
cd load_balancing_analysis
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py
```
