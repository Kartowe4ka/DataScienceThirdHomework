# Исследуйте различные техники регуляризации:
# - Без регуляризации
withoutRegConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "linear", "size": 16},
        {"type": "relu"}
    ]
}
# - Только Dropout (разные коэффициенты: 0.1, 0.3, 0.5)
dropoutConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "linear", "size": 16},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
    ]
}
# - Только BatchNorm
batchConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "batch_norm"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "batch_norm"},
        {"type": "linear", "size": 16},
        {"type": "relu"},
        {"type": "batch_norm"},
    ]
}
# - Dropout + BatchNorm
dropBatchConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 256},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.1},
        {"type": "batch_norm"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "batch_norm"},
        {"type": "linear", "size": 16},
        {"type": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "batch_norm"},
    ]
}
#
# Для каждого варианта:
# - Используйте одинаковую архитектуру
# - Сравните финальную точность
"""
Config           | TrainAcc | TestAcc | TrainLoss | TestLoss |
WithoutRegConfig |  0.9792  | 0.9719  |  0.0699   |  0.0949  |
DropoutConfig    |  0.8379  | 0.9703  |  0.4539   |  0.1269  |
batchConfig      |  0.9974  | 0.9790  |  0.0250   |  0.0772  |
dropBatchConfig  |  0.8436  | 0.9739  |  0.4852   |  0.1281  |
"""
# - Проанализируйте стабильность обучения
"""
По таблице можно заметить, что модели с методом Dropout на обучающих наборах ведут себя нестабильно, не получая 
дойстоного Loss, при этом имея относительно хорошие значения на тестовом наборе
"""


