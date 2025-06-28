# Настройки визуализации
PLOT_CONFIG = {
    "rsrp_range": (-140, -40),
    "rsrq_range": (-20, 0),
    "band_colors": {
        "LTE1800": "blue",
        "LTE2100": "red"
    }
}

# Пороговые значения для анализа
ANALYSIS_THRESHOLDS = {
    "good_rsrp": -85,
    "good_rsrq": -10,
    "coverage_threshold": -100
}