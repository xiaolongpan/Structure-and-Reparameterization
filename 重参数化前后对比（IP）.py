import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import logging
import time  # 新增时间模块
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score  # 新增

# configure the logging format
logging.basicConfig(
    level=logging.INFO,  # You can also use logging.DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


# Load Indian Pines Dataset
def load_indian_pines(data_path, labels_path):
    data = loadmat(data_path)['indian_pines_corrected']  # HSI Data
    labels = loadmat(labels_path)['indian_pines_gt']  # Ground Truth
    return data, labels


# Custom Dataset Class
class IndianPinesDataset(Dataset):
    def __init__(self, data, labels, patch_size=5):
        self.data = data
        self.labels = labels
        self.patch_size = patch_size
        self.padded_data = np.pad(data,
                                  ((patch_size // 2, patch_size // 2), (patch_size // 2, patch_size // 2), (0, 0)),
                                  mode='constant')
        self.samples = self._create_samples()

    def _create_samples(self):
        samples = []
        for i in range(self.labels.shape[0]):
            for j in range(self.labels.shape[1]):
                if self.labels[i, j] > 0:
                    patch = self.padded_data[i:i + self.patch_size, j:j + self.patch_size, :]
                    samples.append((patch, self.labels[i, j] - 1))  # Convert labels to zero-based index
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, label = self.samples[idx]
        patch = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        return patch, torch.tensor(label, dtype=torch.long)


# Train-Test Split
def prepare_dataloaders(data_path, labels_path, batch_size=32):
    data, labels = load_indian_pines(data_path, labels_path)
    # Example Normalization
    data = (data - np.mean(data)) / np.std(data)
    dataset = IndianPinesDataset(data, labels)
    train_size = int(0.7 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 新增：3D版本卷积与BN辅助函数
def conv_bn3d(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
    return result


class RepVGGBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock3d, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.nonlinearity = nn.ReLU()

        self.kernel_size = kernel_size

        if deploy:
            self.rbr_reparam = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups,
                                         bias=True, padding_mode=padding_mode)
        else:
            self.rbr_identity = nn.BatchNorm3d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn3d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn3d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=1, stride=stride, padding=0, groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))
        id_out = self.rbr_identity(inputs) if self.rbr_identity is not None else 0
        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
        # return kernel3x3 + kernelid, bias3x3 + biasid

    def _pad_1x1_to_3x3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            if self.kernel_size == (7, 1, 1):
                return torch.nn.functional.pad(kernel1x1, [0, 0, 0, 0, 3, 3])
            elif self.kernel_size == (1, 3, 3):
                return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1, 0, 0])
            else:
                return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm3d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv3d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class SSTN(nn.Module):
    def __init__(self, input_channels=1, spectral_bands=30, patch_size=5, num_classes=16):
        super(SSTN, self).__init__()
        self.patch_size = patch_size  # 新增：保存patch_size

        # Spectral (1D) convolution
        self.spectral_conv = nn.Conv3d(1, 24, kernel_size=(7, 1, 1), padding=(3, 0, 0))
        self.spectral_bn1 = nn.BatchNorm3d(24)

        # Spatial (2D) convolution
        self.spatial_conv = nn.Conv3d(24, 24, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.spatial_bn1 = nn.BatchNorm3d(24)

        # 动态计算flatten维度
        with torch.no_grad():
            test_input = torch.randn(1, 1, spectral_bands, patch_size, patch_size)
            test_output = self.forward_features(test_input)
            flatten_dim = test_output.view(-1).shape[0]

        self.fc = nn.Linear(flatten_dim, num_classes)

    def forward_features(self, x):
        x = F.relu(self.spectral_bn1(self.spectral_conv(x)))
        x = F.relu(self.spatial_bn1(self.spatial_conv(x)))
        x = torch.mean(x, dim=2)  # [B, C, H, W]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class RepVGGStyleSSTN(SSTN):
    def __init__(self, in_channels=1, patch_size=5, num_classes=16):
        super().__init__(
            input_channels=in_channels,
            spectral_bands=30,  # 根据数据集实际光谱带数量修改
            patch_size=patch_size,
            num_classes=num_classes
        )
        self.reparameterized = False
        self.fused_conv = None

        # 用6个卷积层替换原有的两层卷积
        self.conv1 = RepVGGBlock3d(1, 24, kernel_size=3, stride=1, padding=1)
        self.conv2 = RepVGGBlock3d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv3 = RepVGGBlock3d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv4 = RepVGGBlock3d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv5 = RepVGGBlock3d(24, 24, kernel_size=3, stride=1, padding=1)
        self.conv6 = RepVGGBlock3d(24, 24, kernel_size=3, stride=1, padding=1)

        # Flatten and classification
        flatten_dim = 24 * patch_size * patch_size
        self.fc = nn.Linear(flatten_dim, num_classes)

        # Placeholder for inference
        self.reparameterized = False

    def forward(self, x):
        # 顺序通过6个卷积层，并在光谱维度（dim=2）上做全局平均
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.mean(x, dim=2)  # Global average over spectral dimension
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def repvgg_reparameterize(self):
        # 对所有6个卷积层切换到 deploy 模式
        self.conv1.switch_to_deploy()
        self.conv2.switch_to_deploy()
        self.conv3.switch_to_deploy()
        self.conv4.switch_to_deploy()
        self.conv5.switch_to_deploy()
        self.conv6.switch_to_deploy()


# Training Function
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            train_loss_history.append(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # 改进的损失曲线绘制
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, 'b-', alpha=0.7, label='Batch Loss')

    # 计算并绘制滑动平均（更平滑的曲线）
    window_size = max(1, len(train_loss_history) // 20)  # 自适应窗口大小
    if window_size > 1:
        smooth_loss = np.convolve(train_loss_history, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.arange(window_size - 1, len(train_loss_history)),
                 smooth_loss, 'r-', linewidth=2, label=f'Smoothed ({window_size} batches)')

    plt.title('Training Loss Progress(IN)')
    plt.xlabel('Batch Index')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig('training_loss(IN).png', dpi=300, bbox_inches='tight')
    plt.show()


# Evaluate
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    y_true = []  # 存储真实标签
    y_pred = []  # 存储预测标签
    eval_start = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    eval_time = time.time() - eval_start

    # 计算总体准确率（OA）
    oa = correct / total

    # 计算混淆矩阵和类平均准确率（AA）
    cm = confusion_matrix(y_true, y_pred)
    class_acc = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    aa = np.nanmean(class_acc)

    # 计算卡帕系数（Kappa）
    kappa = cohen_kappa_score(y_true, y_pred)

    # 计算模型参数量（Param）
    params = sum(p.numel() for p in model.parameters())

    print("\nEvaluation metrics:")
    print(f"Overall Accuracy (OA): {100 * oa:.2f}%")
    print(f"Average Accuracy (AA): {100 * aa:.2f}%")
    print(f"Kappa Coefficient: {kappa:.4f}")
    print(f"Parameters (Param): {params}")
    print(f"Evaluation time: {eval_time:.2f}s")
    # 输出每个类别的准确率
    for idx, acc in enumerate(class_acc):
        print(f"Class {idx + 1} Accuracy: {100 * acc:.2f}%")

    # 新增：绘制并保存混淆矩阵热图
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, [str(i + 1) for i in range(len(cm))], rotation=45)
    plt.yticks(tick_marks, [str(i + 1) for i in range(len(cm))])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # 在每个格子里写上数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig('confusion_matrix(IP).png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix heatmap saved as confusion_matrix.png")
    return oa, aa, kappa, params


def visualize_classification_map(model, data_path, labels_path, patch_size=5, save_path='classification_map.png'):
    """
    可视化整幅高光谱图像的分类结果
    """
    model.eval()
    # 读取原始数据和标签
    data = loadmat(data_path)['indian_pines_corrected']
    labels = loadmat(labels_path)['indian_pines_gt']
    h, w, c = data.shape
    # 数据归一化
    data = (data - np.mean(data)) / np.std(data)
    # 填充
    pad = patch_size // 2
    padded = np.pad(data, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    pred_map = np.zeros((h, w), dtype=np.int32)
    device_ = next(model.parameters()).device

    with torch.no_grad():
        for i in range(h):
            for j in range(w):
                if labels[i, j] == 0:
                    continue  # 跳过无标签像素
                patch = padded[i:i + patch_size, j:j + patch_size, :]
                patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(
                    device_)
                output = model(patch_tensor)
                pred = torch.argmax(output, dim=1).item() + 1  # 类别从1开始
                pred_map[i, j] = pred

    # 可视化
    plt.figure(figsize=(8, 8))
    cmap = plt.get_cmap('tab20', np.max(pred_map))
    plt.imshow(pred_map, cmap=cmap, vmin=1, vmax=np.max(pred_map))
    plt.title('Classification Map')
    plt.axis('off')
    cbar = plt.colorbar(ticks=range(1, np.max(pred_map) + 1), fraction=0.046, pad=0.04)
    cbar.set_label('Class')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Classification map saved as {save_path}")


if __name__ == "__main__":
    data_path = r"./datasets/IN/Indian_pines_corrected.mat"
    labels_path = r"./datasets/IN/Indian_pines_gt.mat"
    train_loader, test_loader = prepare_dataloaders(data_path, labels_path)

    # Create model (before reparameterization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RepVGGStyleSSTN(in_channels=1, patch_size=5, num_classes=16).to(device)
    # model = ReparamSSRN(in_channels=1, num_classes=16)

    # Train the model
    train_model(model, train_loader, test_loader, epochs=200, lr=0.001)

    # Track the best validation accuracy during training:
    oa_before, aa_before, kappa_before, params = evaluate(model, test_loader)
    logging.info(f'Before reparam - OA: {oa_before:.4f}, AA: {aa_before:.4f}, Kappa: {kappa_before:.4f}')
    print(f"Metrics before reparameterization - OA: {oa_before:.4f}, AA: {aa_before:.4f}, Kappa: {kappa_before:.4f}")

    # 新增：可视化分类结果
    visualize_classification_map(model, data_path, labels_path, patch_size=5,
                                 save_path='classification_map_before(IP).png')

    # Reparameterize the model
    model.repvgg_reparameterize()

    # Evaluate the reparameterized model
    oa_after, aa_after, kappa_after, _ = evaluate(model, test_loader)
    print(f"Metrics after reparameterization - OA: {oa_after:.4f}, AA: {aa_after:.4f}, Kappa: {kappa_after:.4f}")
    logging.info(f'After reparam - OA: {oa_after:.4f}, AA: {aa_after:.4f}, Kappa: {kappa_after:.4f}')

    # 新增：可视化重参数化后分类结果
    visualize_classification_map(model, data_path, labels_path, patch_size=5,
                                 save_path='classification_map_after(IP).png')

    # Step 6: Save the reparameterized model (optional)
    torch.save(model.state_dict(), 'repvggstyle_ssrn_reparam(IN).pth')
