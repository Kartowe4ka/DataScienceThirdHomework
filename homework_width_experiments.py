# Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
firstWidthConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 64},
        {"type": "relu"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "linear", "size": 16},
        {"type": "relu"}
    ]
}
# - Средние слои: [256, 128, 64]
secondWidthConfig = {
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
# - Широкие слои: [1024, 512, 256]
thirdWidthConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 1024},
        {"type": "relu"},
        {"type": "linear", "size": 512},
        {"type": "relu"},
        {"type": "linear", "size": 256},
        {"type": "relu"}
    ]
}
# - Очень широкие слои: [2048, 1024, 512]
fourthWidthConfig = {
    "input_size": 784,
    "num_classes": 10,
    "layers": [
        {"type": "linear", "size": 2048},
        {"type": "relu"},
        {"type": "linear", "size": 1024},
        {"type": "relu"},
        {"type": "linear", "size": 512},
        {"type": "relu"}
    ]
}
#
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров
"""
Config         | TrainAcc | TestAcc | TrainLoss | TestLoss | TimeSpend (с) | Кол-во параметров
first_config   |  0.9613  | 0.9592  |  0.1214   |  0.1327  |      52       |       53018
second_config  |  0.9795  | 0.9703  |  0.0691   |  0.0944  |      60       |       209882
third_config   |  0.9964  | 0.9823  |  0.0104   |  0.0778  |      98       |       1462538
fourth_config  |  0.9970  | 0.9811  |  0.0089   |  0.0773  |      93       |       4235786
"""
