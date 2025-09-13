# DCGAN 生成手写数字 Demo
# FishCat233@github.com
# 2025-04-26 17:02:21
import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


class PrintShape(torch.nn.Module):
    def __init__(self, text: str = " "):
        super().__init__()
        self.text = text

    def forward(self, x):
        print(self.text, x.shape, "\n")
        if x.shape[2] == 8:
            print("")
        return x


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            # conv shape calc: size = (size + 2 padding - kernel_size) / stride + 1
            # PrintShape("D input "),
            nn.Conv2d(1, 64, 3, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            # PrintShape("D tst "),
            nn.Conv2d(256, 1, 8, 1, 0),
            # PrintShape("D output "),  # batch_size, 1, 3, 3
            torch.nn.Sigmoid(),  # 输出 0~1 之间
        )

    def forward(self, x):
        x = self.model(x)

        return x


class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            # conv transpose shape calc: size = (size - 1) * stride - 2 * padding + kernel_size
            # PrintShape("G input "),
            nn.ConvTranspose2d(100, 1024, 4, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.ReLU(),
            # PrintShape("G tst"),
            nn.ConvTranspose2d(128, 1, 2, 2, 2),
            # PrintShape(), # 10,1,32,32
            # PrintShape("G output "),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = x.reshape(-1, 100, 1, 1)
        x: torch.Tensor = self.model(x)

        return x


def train(epochs=100, model_G=None, model_D=None, pre_epoch_num=0):
    # 数据集导入
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])

    train_data = MNIST("./data", transform=transformer, download=True)

    dataloader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    # 模型导入
    D = Discriminator().to(device) if model_D is None else model_D.to(device)
    G = Generator().to(device) if model_G is None else model_G.to(device)

    D_optim = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    G_optim = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))

    loss_fn = torch.nn.BCELoss()

    # 训练
    for epoch in range(epochs):
        dis_loss_all = 0
        gen_loss_all = 0
        loader_len = len(dataloader)
        for step, data in tqdm(
            enumerate(dataloader), desc=f"Epoch {epoch + pre_epoch_num + 1} / {epochs}", total=loader_len
        ):
            # 判别器损失
            # sample shape: [batch_size, 1, 28, 28]
            sample, _ = data
            sample = sample.reshape(-1, 1, 28, 28)
            sample: torch.Tensor = sample.to(device)

            # 正态分布采样
            sample_z = torch.normal(0, 1, size=(sample.shape[0], 100), device=device)

            # 计算损失
            y = D(sample)
            true_loss = loss_fn(y, torch.ones_like(y))

            y = D(G(sample_z.detach()))
            fake_loss = loss_fn(y, torch.zeros_like(y))

            # 更新
            # if epoch % 2 == 0:
            dis_loss = true_loss + fake_loss
            D_optim.zero_grad()
            # if dis_loss_all / (gen_loss_all + 1e-7) < 1 / 2:
            #     dis_loss.backward()
            #     D_optim.step()

            dis_loss.backward()
            D_optim.step()

            # 生成器损失
            y = D(G(sample_z))
            gen_loss = loss_fn(y, torch.ones_like(y))

            G_optim.zero_grad()
            # if dis_loss_all / (gen_loss_all + 1e-7) > 1:
            #     gen_loss.backward()
            #     G_optim.step()

            gen_loss.backward()
            G_optim.step()

            with torch.no_grad():
                dis_loss_all += dis_loss.item()
                gen_loss_all += gen_loss.item()

        with torch.no_grad():
            # 计算当前轮损失
            dis_loss_all = dis_loss_all / loader_len
            gen_loss_all = gen_loss_all / loader_len
            print(f"判别器损失 {dis_loss_all: .6f} | 生成器损失 {gen_loss_all: .6f}")

        os.makedirs(f"model/epoch_{epoch + pre_epoch_num}", exist_ok=True)
        torch.save(G, f"model/epoch_{epoch + pre_epoch_num}/G_{dis_loss_all}.pth")
        torch.save(D, f"model/epoch_{epoch + pre_epoch_num}/D_{gen_loss_all}.pth")

    # 保存训练最后的模型
    torch.save(G, f"model/G.pth")
    torch.save(D, f"model/D.pth")

    # 记录训练结束的 epoch
    with open("model/epoch.txt", "w") as f:
        f.write(str(epochs + pre_epoch_num))


# if __name__ == '__main__':
#     model_G = Discriminator().to(device)
#
#     # x = torch.normal(0, 1, size=(10, 1600), device=device)
#     x = torch.normal(0, 1, size=(10, 1, 28, 28), device=device)
#
#     y = model_G(x)

if __name__ == '__main__':
    # 训练
    model_G = None
    model_D = None
    pre_epochs = 0
    if os.path.exists("model/G.pth") and os.path.exists("model/D.pth"):
        model_G = torch.load("model/G.pth")
        model_D = torch.load("model/D.pth")

        with open("model/epoch.txt", "r") as f:
            pre_epochs = int(f.read())
    train(epochs=25, model_G=None, model_D=None, pre_epoch_num=pre_epochs)

    # 生成
    if model_G is None:
        model_G = torch.load("model/G.pth")
    if model_D is None:
        model_D = torch.load("model/D.pth")

    model_G.eval()
    model_D.eval()

    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    train_data = MNIST("./data", transform=transformer, download=True)

    # 创建一个10张图片的图表
    plt.figure(figsize=(12, 6))

    with torch.no_grad():
        # 生成 5 张假图片
        fake_z = torch.normal(0, 1, size=(5, 100), device=device)
        fake_images = model_G(fake_z)
        fake_scores = model_D(fake_images).cpu().numpy().flatten()
        fake_images = fake_images.detach().cpu().numpy()

        # 绘制假图片
        for i in range(5):
            plt.subplot(2, 5, i + 1)
            plt.imshow(fake_images[i].squeeze(), cmap='gray')
            plt.title(f"Fake\nScore: {fake_scores[i]:.4f}")
            plt.axis('off')

        # 获取 5 张真图片
        real_images = []
        for i in range(5):
            real_images.append(train_data[i][0])
        real_images_tensor = torch.stack(real_images).to(device)
        real_scores = model_D(real_images_tensor).cpu().numpy().flatten()

        # 绘制真图片
        for i in range(5):
            plt.subplot(2, 5, i + 6)
            plt.imshow(real_images[i].squeeze(), cmap='gray')
            plt.title(f"Real\nScore: {real_scores[i]:.4f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

    fake_z = torch.normal(0, 1, size=(10, 100), device=device)
    y = model_G(fake_z)
    y: torch.Tensor = y.detach()
    y = y.squeeze(1).to(cpu).numpy()

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(y[i], cmap='gray')

    plt.show()
