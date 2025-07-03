# Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
first_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": []
    }
# - 2 слоя (1 скрытый)
second_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 128},
            {"type": "relu"},
        ]
    }
# - 3 слоя (2 скрытых)
third_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"},
        ]
    }
# - 5 слоев (4 скрытых)
fourth_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    }
# - 7 слоев (6 скрытых)
fifth_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 2048},
            {"type": "relu"},
            {"type": "linear", "size": 1024},
            {"type": "relu"},
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    }
#
# Для каждого варианта:
# - Сравните точность на train и test
# - Проанализируйте время обучения
"""
Config         | TrainAcc | TestAcc | TrainLoss | TestLoss | TimeSpend (с)
first_config   |  0.9224  | 0.9222  |  0.2772   |  0.2731  |      93
second_config  |  0.9790  | 0.9708  |  0.0768   |  0.0951  |      65
third_config   |  0.9898  | 0.9778  |  0.0358   |  0.0722  |      77
fourth_config  |  0.9957  | 0.9759  |  0.0142   |  0.0846  |      89
fifth_config   |  0.9954  | 0.9807  |  0.0143   |  0.0775  |      118
"""
# - Визуализируйте кривые обучения
# Все графики будут находиться в папке plots

# Исследуйте влияние глубины на переобучение:
# Чем больше глубина - тем вероятнее переобучение

# - Постройте графики train/test accuracy по эпохам
# Все графики будут находиться в папке plots

# - Определите оптимальную глубину для каждого датасета
"""
MNISTDataset - fourth_config
CIFARDataset - third_config
"""

# - Добавьте Dropout и BatchNorm, сравните результаты
# Проведем сравнение с second_config:
secondDropout_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 128},
            {"type": "relu"},
            {"type": "dropout", "rate": 0.3}
        ]
    }

secondBatch_config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 128},
            {"type": "relu"},
            {"type": "batch_norm"}
        ]
    }

"""
Config               | TrainAcc | TestAcc | TrainLoss | TestLoss | TimeSpend (с)
second_config        |  0.9790  | 0.9708  |  0.0768   |  0.0951  |      65
secondDropout_config |  0.9649  | 0.9720  |  0.1185   |  0.0973  |      56
secondBatch_config   |  0.9922  | 0.9724  |  0.0362   |  0.0882  |      72
"""

# - Проанализируйте, когда начинается переобучение
# Переобучение начинается после 8 эпохи
