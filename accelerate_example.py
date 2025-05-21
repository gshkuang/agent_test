import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
# 1. 初始化Accelerator对象，自动检测并分配设备（CPU/GPU/TPU）
accelerator = Accelerator()
print(f"Using device: {accelerator.device}")


# 2. 构建一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(x)


# 3. 构造简单的数据集
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 4. 实例化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. 使用accelerator准备模型、优化器和数据加载器
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# 6. 训练循环
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        # 7. 使用accelerator.backward替代loss.backward，兼容混合精度和分布式
        accelerator.backward(loss)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# 8. 保存模型时，需用accelerator.unwrap_model获取原始模型
unwrapped_model = accelerator.unwrap_model(model)
torch.save(unwrapped_model.state_dict(), "simple_net.pt")

# 主要原理说明：
# - Accelerator自动处理多卡/混合精度/分布式训练，无需手动写.cuda()或DDP相关代码。
# - accelerator.prepare会自动将模型、优化器、数据加载器迁移到合适设备，并封装分布式/混合精度逻辑。
# - accelerator.backward兼容混合精度和分布式反向传播。
# - unwrap_model用于保存原始模型权重，避免保存分布式封装后的模型。
# - 通过accelerate config/config.yaml可配置DeepSpeed、FP16等高级加速策略。

# ========== DeepSpeed 集成说明 ==========
# 1. 先安装 deepspeed: pip install deepspeed
# 2. 使用 accelerate config 生成配置文件，选择 DeepSpeed 相关选项
# 3. 运行脚本时使用 accelerate launch --config_file=your_config.yaml accelerate_example.py
# 4. 你可以在 accelerate 配置文件中指定 DeepSpeed 配置（如零冗余优化、梯度累积等）
# 5. 训练脚本无需显式引入 deepspeed，accelerate 会自动集成
# 6. 典型命令行示例：
#    accelerate launch --config_file=your_config.yaml accelerate_example.py
# =======================================
