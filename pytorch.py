import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ネットワークの定義
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 784)
        x = self.fc(x)
        return x


def main():
    # GPUが利用可能かどうかをチェック
    gpu_id = 1  # set manually
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"{device=}")

    # データの前処理
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # MNISTデータセットの読み込み
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    """
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    """

    # データローダーの設定
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ネットワークのインスタンス化
    net = Net().to(device)

    # 損失関数と最適化アルゴリズムの設定
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # 学習ループ
    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            data: tuple[torch.Tensor, torch.Tensor]
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs: torch.Tensor = net(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print("Finished training")
    return


if __name__ == "__main__":
    main()
