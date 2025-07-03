import time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.model_utils import FullyConnectedModel as FCN
from utils.datasets_utils import get_mnist_loaders, get_cifar_loaders
from visualization_utils import getLearningCurve, getAccuracyCurve
from homework_depth_experiments import first_config, second_config, third_config, fourth_config, fifth_config
from homework_depth_experiments import secondDropout_config, secondBatch_config
from homework_width_experiments import firstWidthConfig, secondWidthConfig, thirdWidthConfig, fourthWidthConfig
from homework_regularization_experiments import withoutRegConfig, dropoutConfig, batchConfig, dropBatchConfig


def run_epoch(model, data_loader, criterion, optimizer=None, device='cpu', is_test=False):
    if is_test:
        model.eval()
    else:
        model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = data.to(device), target.to(device)

        if not is_test and optimizer is not None:
            optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        if not is_test and optimizer is not None:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    return total_loss / len(data_loader), correct / total


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    train_losses_plt, train_accs_plt = [], []
    test_losses_plt, test_accs_plt = [], []
    epochs_plt = [i for i in range(1, 10 + 1)]

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_test=False)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device, is_test=True)
        train_losses_plt.append(train_loss)
        train_accs_plt.append(train_acc)
        test_losses_plt.append(test_loss)
        test_accs_plt.append(test_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        print('-' * 50)

    getLearningCurve(epochs_plt, train_losses_plt, train_accs_plt, "Loss и Acc на Train")
    getLearningCurve(epochs_plt, test_losses_plt, test_accs_plt, "Loss и Acc на Test")
    getAccuracyCurve(epochs_plt, train_accs_plt, "Acc на Train")
    getAccuracyCurve(epochs_plt, test_accs_plt, "Acc на Test")

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs
    }


if __name__ == "__main__":
    config = {
        "input_size": 784,
        "num_classes": 10,
        "layers": [
            {"type": "linear", "size": 512},
            {"type": "relu"},
            {"type": "linear", "size": 256},
            {"type": "relu"},
            {"type": "linear", "size": 128},
            {"type": "relu"}
        ]
    }

    model = FCN(**secondBatch_config)
    print(model)
    print(f"Model params: {count_parameters(model)}")

    train_d1, test_d1 = get_mnist_loaders(batch_size=1024)
    start = time.time()
    train_model(model, train_d1, test_d1, epochs=10, lr=0.001, device="cuda")
    end = time.time()
    print(end - start)
