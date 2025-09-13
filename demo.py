from main import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")

if __name__ == '__main__':
    # 训练
    model_G = None
    model_D = None

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
        # 生成 10 张假图片
        fake_z = torch.normal(0, 1, size=(10, 100), device=device)
        fake_images = model_G(fake_z)
        fake_scores = model_D(fake_images).cpu().numpy().flatten()
        fake_images = fake_images.detach().cpu().numpy()

        # 绘制假图片
        for i in range(10):
            plt.subplot(2, 10, i + 1)
            plt.imshow(fake_images[i].squeeze(), cmap='gray')
            plt.title(f"Fake\nScore: {fake_scores[i]:.4f}")
            plt.axis('off')

        # 获取 10 张真图片
        real_images = []
        for i in range(10):
            real_images.append(train_data[i][0])
        real_images_tensor = torch.stack(real_images).to(device)
        real_scores = model_D(real_images_tensor).cpu().numpy().flatten()

        # 绘制真图片
        for i in range(10):
            plt.subplot(2, 10, i + 11)
            plt.imshow(real_images[i].squeeze(), cmap='gray')
            plt.title(f"Real\nScore: {real_scores[i]:.4f}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()
