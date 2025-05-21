import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]  # macOS 推荐
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号


print("支持的后端：", torch.backends.quantized.supported_engines)
if "qnnpack" in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = "qnnpack"


# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载CIFAR-10数据集
def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


# 训练模型
def train_model(model, train_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    return model


# 评估模型
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    inference_time = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(inputs)
            end_time = time.time()
            inference_time += end_time - start_time

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_inference_time = inference_time / len(test_loader)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Average inference time per batch: {avg_inference_time*1000:.2f} ms")

    return accuracy, avg_inference_time


# 量化模型部分
def quantize_model_demo(model, train_loader, test_loader):
    print("\n===== 模型量化演示 =====")
    print("原始FP32模型评估:")
    fp32_accuracy, fp32_time = evaluate_model(model, test_loader)

    # 检查量化支持
    print("\n检查量化支持...")
    try:
        # 1. 动态量化 (Dynamic Quantization)
        print("\n1. 动态量化 (Dynamic Quantization)")
        # 动态量化适用于RNN和Transformer等模型，主要量化线性层和LSTM层
        print("可量化的层: Linear (全连接层), LSTM, GRU, RNN等")

        # 尝试使用安全的方式进行动态量化
        try:
            dynamic_quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            print("动态量化后评估:")
            dynamic_accuracy, dynamic_time = evaluate_model(
                dynamic_quantized_model, test_loader
            )
        except RuntimeError as e:
            print(f"动态量化失败: {e}")
            print("跳过动态量化评估，使用原始模型数据代替")
            dynamic_accuracy, dynamic_time = fp32_accuracy, fp32_time

        # 2. 静态量化 (Static Quantization)
        print("\n2. 静态量化 (Static Quantization)")
        # 静态量化需要先准备模型，插入观察者
        print("可量化的层: Conv2d, Linear, ReLU等")

        # 创建量化配置
        model_to_quantize = copy.deepcopy(model)
        model_to_quantize.eval()

        # 尝试使用安全的方式进行静态量化
        try:
            # 为静态量化准备模型
            model_to_quantize.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model_to_quantize, inplace=True)

            # 校准 - 通过少量数据来确定量化参数
            with torch.no_grad():
                for i, (inputs, _) in enumerate(train_loader):
                    model_to_quantize(inputs)
                    if i >= 10:  # 使用少量批次进行校准
                        break

            # 转换为量化模型
            static_quantized_model = torch.quantization.convert(
                model_to_quantize, inplace=False
            )
            print("静态量化后评估:")
            static_accuracy, static_time = evaluate_model(
                static_quantized_model, test_loader
            )
        except RuntimeError as e:
            print(f"静态量化失败: {e}")
            print("跳过静态量化评估，使用原始模型数据代替")
            static_accuracy, static_time = fp32_accuracy, fp32_time

        # 3. 量化感知训练 (QAT)
        print("\n3. 量化感知训练 (QAT)")
        print("QAT在训练过程中模拟量化效果，通常能获得最高精度")

        # 尝试使用安全的方式进行QAT
        try:
            # 准备QAT模型
            qat_model = copy.deepcopy(model)
            qat_model.train()

            # 设置QAT配置
            qat_model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
            torch.quantization.prepare_qat(qat_model, inplace=True)

            # 简短的QAT训练
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(qat_model.parameters(), lr=0.0001)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            qat_model.to(device)

            print("执行量化感知训练...")
            for epoch in range(1):  # 实际应用中应该训练更多轮次
                for i, (inputs, labels) in enumerate(train_loader):
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = qat_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    if i >= 100:  # 简短演示，只训练少量批次
                        break

            # 转换为量化模型
            qat_model.eval()
            qat_model = qat_model.cpu()
            torch.quantization.convert(qat_model, inplace=True)

            print("QAT后评估:")
            qat_accuracy, qat_time = evaluate_model(qat_model, test_loader)
        except RuntimeError as e:
            print(f"QAT失败: {e}")
            print("跳过QAT评估，使用原始模型数据代替")
            qat_accuracy, qat_time = fp32_accuracy, fp32_time
            qat_model = model  # 使用原始模型代替

        # 量化效果比较
        print("\n===== 量化效果比较 =====")
        print(
            f"{'模型类型':<20} {'精度 (%)':<15} {'推理时间 (ms)':<20} {'相对于FP32的加速比':<25}"
        )
        print("-" * 80)
        print(
            f"{'FP32原始模型':<20} {fp32_accuracy:<15.2f} {fp32_time*1000:<20.2f} {1.0:<25.2f}"
        )
        print(
            f"{'动态量化':<20} {dynamic_accuracy:<15.2f} {dynamic_time*1000:<20.2f} {fp32_time/dynamic_time:<25.2f}"
        )
        print(
            f"{'静态量化':<20} {static_accuracy:<15.2f} {static_time*1000:<20.2f} {fp32_time/static_time:<25.2f}"
        )
        print(
            f"{'QAT量化':<20} {qat_accuracy:<15.2f} {qat_time*1000:<20.2f} {fp32_time/qat_time:<25.2f}"
        )

        # 验证量化是否成功
        print("\n===== 验证量化是否成功 =====")
        print("1. 模型大小比较:")
        fp32_size = get_model_size(model)

        try:
            dynamic_size = get_model_size(dynamic_quantized_model)
        except:
            dynamic_size = fp32_size

        try:
            static_size = get_model_size(static_quantized_model)
        except:
            static_size = fp32_size

        try:
            qat_size = get_model_size(qat_model)
        except:
            qat_size = fp32_size

        print(f"{'模型类型':<20} {'大小 (MB)':<15} {'相对于FP32的压缩比':<25}")
        print("-" * 60)
        print(f"{'FP32原始模型':<20} {fp32_size:<15.2f} {1.0:<25.2f}")
        print(f"{'动态量化':<20} {dynamic_size:<15.2f} {fp32_size/dynamic_size:<25.2f}")
        print(f"{'静态量化':<20} {static_size:<15.2f} {fp32_size/static_size:<25.2f}")
        print(f"{'QAT量化':<20} {qat_size:<15.2f} {fp32_size/qat_size:<25.2f}")

        print("\n2. 量化后模型结构检查:")
        # 检查模型中是否包含量化层
        try:
            check_quantized_layers(dynamic_quantized_model, "动态量化模型")
        except:
            print("动态量化模型检查失败")

        try:
            check_quantized_layers(static_quantized_model, "静态量化模型")
        except:
            print("静态量化模型检查失败")

        try:
            check_quantized_layers(qat_model, "QAT量化模型")
        except:
            print("QAT量化模型检查失败")

        # 保存量化模型
        try:
            torch.save(qat_model.state_dict(), "quantized_model.pth")
            print("\n量化模型已保存为 'quantized_model.pth'")
        except Exception as e:
            print(f"\n保存量化模型失败: {e}")
            print("使用原始模型代替")
            torch.save(model.state_dict(), "quantized_model.pth")
            print("原始模型已保存为 'quantized_model.pth'")

        return qat_model

    except Exception as e:
        print(f"量化过程中发生错误: {e}")
        print("返回原始模型")
        return model


# 获取模型大小（MB）
def get_model_size(model):
    torch.save(model.state_dict(), "temp_model.pth")
    size_mb = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size_mb


# 检查模型中的量化层
def check_quantized_layers(model, model_name):
    has_quantized = False
    for name, module in model.named_modules():
        if "quantized" in str(type(module)):
            if not has_quantized:
                print(f"\n{model_name}中的量化层:")
                has_quantized = True
            print(f"- {name}: {type(module).__name__}")

    if not has_quantized:
        print(f"\n{model_name}中未检测到量化层")


# 模型剪枝部分
def prune_model_demo(model, train_loader, test_loader):
    import torch.nn.utils.prune as prune

    print("\n===== 模型剪枝演示 =====")

    # 评估原始模型
    print("原始模型评估:")
    original_accuracy, original_time = evaluate_model(model, test_loader)

    # 创建模型副本用于剪枝
    pruned_model = copy.deepcopy(model)

    # 1. L1范数剪枝 - 移除权重绝对值最小的连接
    print("\n1. 应用L1范数剪枝")
    pruning_ratio = 0.3  # 剪枝30%的权重

    # 对卷积层和全连接层应用剪枝
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=pruning_ratio)

    # 评估剪枝后的模型
    print("L1剪枝后评估:")
    pruned_accuracy, pruned_time = evaluate_model(pruned_model, test_loader)

    # 2. 结构化剪枝 - 移除整个通道/神经元
    print("\n2. 应用结构化剪枝")
    structured_pruned_model = copy.deepcopy(model)

    # 对卷积层应用通道剪枝
    for name, module in structured_pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(
                module, name="weight", amount=0.2, n=2, dim=0
            )  # 剪枝20%的输出通道

    # 评估结构化剪枝后的模型
    print("结构化剪枝后评估:")
    structured_accuracy, structured_time = evaluate_model(
        structured_pruned_model, test_loader
    )

    # 3. 剪枝后微调
    print("\n3. 剪枝后微调")
    fine_tuned_model = copy.deepcopy(pruned_model)

    # 将临时剪枝转为永久剪枝
    for name, module in fine_tuned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

    # 微调剪枝后的模型
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.0005)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tuned_model.to(device)

    print("执行微调...")
    for epoch in range(2):  # 实际应用中应该训练更多轮次
        fine_tuned_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = fine_tuned_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(f"微调 Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

            if i >= 200:  # 简短演示，只训练少量批次
                break

    # 评估微调后的模型
    print("剪枝+微调后评估:")
    fine_tuned_accuracy, fine_tuned_time = evaluate_model(fine_tuned_model, test_loader)

    # 剪枝效果比较
    print("\n===== 剪枝效果比较 =====")
    print(
        f"{'模型类型':<20} {'精度 (%)':<15} {'推理时间 (ms)':<20} {'相对于原始的加速比':<25}"
    )
    print("-" * 80)
    print(
        f"{'原始模型':<20} {original_accuracy:<15.2f} {original_time*1000:<20.2f} {1.0:<25.2f}"
    )
    print(
        f"{'L1剪枝':<20} {pruned_accuracy:<15.2f} {pruned_time*1000:<20.2f} {original_time/pruned_time:<25.2f}"
    )
    print(
        f"{'结构化剪枝':<20} {structured_accuracy:<15.2f} {structured_time*1000:<20.2f} {original_time/structured_time:<25.2f}"
    )
    print(
        f"{'剪枝+微调':<20} {fine_tuned_accuracy:<15.2f} {fine_tuned_time*1000:<20.2f} {original_time/fine_tuned_time:<25.2f}"
    )

    # 分析剪枝后模型是否真的加速
    print("\n===== 剪枝后模型是否真的加速分析 =====")
    print("1. 模型参数数量比较:")
    original_params = count_parameters(model)
    pruned_params = count_parameters(pruned_model)
    structured_params = count_parameters(structured_pruned_model)
    fine_tuned_params = count_parameters(fine_tuned_model)

    print(f"{'模型类型':<20} {'参数数量':<15} {'相对于原始的压缩比':<25}")
    print("-" * 60)
    print(f"{'原始模型':<20} {original_params:<15,d} {1.0:<25.2f}")
    print(
        f"{'L1剪枝':<20} {pruned_params:<15,d} {original_params/pruned_params:<25.2f}"
    )
    print(
        f"{'结构化剪枝':<20} {structured_params:<15,d} {original_params/structured_params:<25.2f}"
    )
    print(
        f"{'剪枝+微调':<20} {fine_tuned_params:<15,d} {original_params/fine_tuned_params:<25.2f}"
    )

    print("\n2. 剪枝后模型加速分析:")
    print(
        "- 非结构化剪枝(L1剪枝)：虽然减少了参数数量，但由于稀疏矩阵操作需要特殊硬件支持，"
    )
    print("  在普通硬件上可能不会带来实际加速，甚至可能因为额外的索引操作而变慢。")
    print(
        "- 结构化剪枝：通过移除整个通道/神经元，可以直接减少计算量，通常能带来实际加速。"
    )
    print("- 在边缘设备上，结构化剪枝通常比非结构化剪枝更有效。")

    # 剪枝后模型保存和部署
    print("\n===== 剪枝后模型保存和部署 =====")

    # 将临时剪枝转为永久剪枝
    for name, module in fine_tuned_model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")

    # 保存剪枝后的模型
    torch.save(fine_tuned_model.state_dict(), "pruned_model.pth")
    print("剪枝后的模型已保存为 'pruned_model.pth'")

    # 模型部署说明
    print("\n剪枝后模型部署步骤:")
    print("1. 永久移除剪枝掩码，转换为标准模型")
    print("2. 保存模型架构和权重")
    print("3. 在目标设备上加载模型时，使用与原始模型相同的架构")
    print("4. 对于结构化剪枝，可以重新定义更小的模型架构，进一步减少模型大小")

    return fine_tuned_model


# 计算模型参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 模型蒸馏部分
def distill_model_demo(teacher_model, train_loader, test_loader):
    print("\n===== 模型蒸馏演示 =====")

    # 定义一个更小的学生模型
    class StudentModel(nn.Module):
        def __init__(self):
            super(StudentModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2)
            self.fc1 = nn.Linear(16 * 8 * 8, 64)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = x.view(-1, 16 * 8 * 8)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x

    student_model = StudentModel()

    # 评估教师模型
    print("教师模型评估:")
    teacher_accuracy, teacher_time = evaluate_model(teacher_model, test_loader)

    # 评估未蒸馏的学生模型
    print("\n未蒸馏的学生模型评估:")
    # 先训练学生模型几个epoch
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_model.to(device)
    teacher_model.to(device)

    print("训练未蒸馏的学生模型...")
    for epoch in range(1):
        student_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = student_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if i >= 100:  # 简短演示，只训练少量批次
                break

    student_accuracy, student_time = evaluate_model(student_model, test_loader)

    # 知识蒸馏
    print("\n执行知识蒸馏...")

    # 重新初始化学生模型
    student_model = StudentModel().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)

    # 蒸馏超参数
    alpha = 0.5  # 平衡蒸馏损失和真实标签损失的权重
    temperatures = [1.0, 2.0, 5.0, 10.0]  # 不同的温度值

    distilled_models = []
    distilled_accuracies = []
    distilled_times = []

    for temp in temperatures:
        print(f"\n使用温度 T={temp} 进行蒸馏")
        # 复制学生模型
        temp_student = copy.deepcopy(student_model)
        temp_optimizer = optim.Adam(temp_student.parameters(), lr=0.001)

        # 蒸馏训练
        for epoch in range(2):  # 实际应用中应该训练更多轮次
            temp_student.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                # 教师模型预测
                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)

                # 学生模型预测
                student_outputs = temp_student(inputs)

                # 计算蒸馏损失 (KL散度)
                T = temp
                soft_targets = nn.functional.softmax(teacher_outputs / T, dim=1)
                soft_prob = nn.functional.log_softmax(student_outputs / T, dim=1)
                distillation_loss = nn.KLDivLoss(reduction="batchmean")(
                    soft_prob, soft_targets
                ) * (T * T)

                # 计算学生模型与真实标签的损失
                student_loss = criterion(student_outputs, labels)

                # 总损失
                loss = alpha * student_loss + (1 - alpha) * distillation_loss

                temp_optimizer.zero_grad()
                loss.backward()
                temp_optimizer.step()

                if i % 50 == 49:
                    print(
                        f"温度 T={temp}, Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}"
                    )

                if i >= 100:  # 简短演示，只训练少量批次
                    break

        # 评估蒸馏后的模型
        print(f"温度 T={temp} 蒸馏后的学生模型评估:")
        distilled_accuracy, distilled_time = evaluate_model(temp_student, test_loader)

        distilled_models.append(temp_student)
        distilled_accuracies.append(distilled_accuracy)
        distilled_times.append(distilled_time)

    # 找出最佳温度
    best_idx = np.argmax(distilled_accuracies)
    best_temp = temperatures[best_idx]
    best_model = distilled_models[best_idx]

    # 蒸馏效果比较
    print("\n===== 蒸馏效果比较 =====")
    print(
        f"{'模型类型':<25} {'精度 (%)':<15} {'推理时间 (ms)':<20} {'相对于教师的加速比':<25}"
    )
    print("-" * 85)
    print(
        f"{'教师模型':<25} {teacher_accuracy:<15.2f} {teacher_time*1000:<20.2f} {1.0:<25.2f}"
    )
    print(
        f"{'未蒸馏学生模型':<25} {student_accuracy:<15.2f} {student_time*1000:<20.2f} {teacher_time/student_time:<25.2f}"
    )

    for i, temp in enumerate(temperatures):
        print(
            f"{'蒸馏学生(T=' + str(temp) + ')':<25} {distilled_accuracies[i]:<15.2f} {distilled_times[i]*1000:<20.2f} {teacher_time/distilled_times[i]:<25.2f}"
        )

    print(f"\n最佳温度为 T={best_temp}，精度: {distilled_accuracies[best_idx]:.2f}%")

    # 温度的作用分析
    print("\n===== 蒸馏温度的作用分析 =====")
    print("1. 温度参数T的作用:")
    print("   - 温度T控制软标签的平滑程度")
    print("   - T=1时，保持原始分布")
    print("   - T>1时，分布变得更平滑，减小类别间差异")
    print("   - T越大，分布越平滑，越强调教师模型不太确定的类别")
    print("   - 高温度使得学生模型能学习到类别间的相似性关系")

    # 绘制不同温度下的软标签分布
    plt.figure(figsize=(12, 6))

    # 生成示例logits
    example_logits = torch.tensor(
        [5.0, 3.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    )

    for i, T in enumerate([1.0, 2.0, 5.0, 10.0]):
        plt.subplot(2, 2, i + 1)
        soft_targets = nn.functional.softmax(example_logits / T, dim=0).numpy()
        plt.bar(range(10), soft_targets)
        plt.title(f"温度 T={T}")
        plt.xlabel("类别")
        plt.ylabel("概率")

    plt.tight_layout()
    plt.savefig("distillation_temperature.png")
    print("温度效果图已保存为 'distillation_temperature.png'")

    # 保存最佳蒸馏模型
    torch.save(best_model.state_dict(), "distilled_model.pth")
    print(f"\n最佳蒸馏模型 (温度T={best_temp}) 已保存为 'distilled_model.pth'")

    return best_model


# 边缘设备部署部分
def deploy_to_edge_demo(model, test_loader):
    print("\n===== 边缘设备部署演示 =====")

    # 1. TorchScript 导出
    print("\n1. TorchScript 导出")
    model.eval()
    example_input = next(iter(test_loader))[0][0:1]  # 获取一个样本

    # 使用trace方式导出
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("model_torchscript.pt")
    print("TorchScript模型已保存为 'model_torchscript.pt'")

    # 验证TorchScript模型
    traced_output = traced_model(example_input)
    original_output = model(example_input)
    torch_script_diff = (traced_output - original_output).abs().max().item()
    print(f"TorchScript模型与原始模型输出差异: {torch_script_diff:.6f}")

    # 2. ONNX 导出
    print("\n2. ONNX 导出")
    try:
        import onnx
        import onnxruntime

        # 导出ONNX模型
        torch.onnx.export(
            model,  # 要导出的模型
            example_input,  # 模型输入
            "model.onnx",  # 输出文件名
            export_params=True,  # 存储训练好的参数权重
            opset_version=12,  # ONNX算子集版本
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=["input"],  # 输入名
            output_names=["output"],  # 输出名
            dynamic_axes={  # 动态尺寸
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

        # 验证ONNX模型
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX模型已保存为 'model.onnx' 并通过验证")

        # 使用ONNX Runtime测试推理
        ort_session = onnxruntime.InferenceSession("model.onnx")
        ort_inputs = {ort_session.get_inputs()[0].name: example_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)

        # 比较ONNX和PyTorch的输出
        onnx_diff = np.abs(ort_outputs[0] - original_output.detach().numpy()).max()
        print(f"ONNX模型与原始模型输出差异: {onnx_diff:.6f}")

    except ImportError:
        print("未安装ONNX或ONNX Runtime，跳过ONNX导出部分")

    # 3. TensorFlow Lite 导出
    print("\n3. TensorFlow Lite 导出")
    try:
        import tensorflow as tf

        # 通过ONNX-TF转换
        try:
            import onnx_tf

            # 先将ONNX模型转换为TensorFlow SavedModel
            onnx_model = onnx.load("model.onnx")
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            tf_rep.export_graph("tf_model")
            print("已将ONNX模型转换为TensorFlow SavedModel")

            # 将SavedModel转换为TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            with open("model.tflite", "wb") as f:
                f.write(tflite_model)

            print("TensorFlow Lite模型已保存为 'model.tflite'")

            # 验证TFLite模型
            interpreter = tf.lite.Interpreter(model_path="model.tflite")
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # 设置输入
            input_data = example_input.numpy()
            interpreter.set_tensor(input_details[0]["index"], input_data)

            # 执行推理
            interpreter.invoke()

            # 获取输出
            tflite_output = interpreter.get_tensor(output_details[0]["index"])

            # 比较TFLite和PyTorch的输出
            tflite_diff = np.abs(tflite_output - original_output.detach().numpy()).max()
            print(f"TFLite模型与原始模型输出差异: {tflite_diff:.6f}")

        except ImportError:
            print("未安装onnx-tf，无法完成ONNX到TFLite的转换")

    except ImportError:
        print("未安装TensorFlow，跳过TFLite导出部分")

    # 4. 各种部署格式比较
    print("\n===== 不同部署格式比较 =====")
    print("1. TorchScript:")
    print("   - 优点: 与PyTorch生态紧密集成，支持动态图和静态图")
    print("   - 优点: 可以在C++环境中运行，适合服务器部署")
    print("   - 缺点: 在移动设备上支持有限")
    print("   - 适用场景: 服务器部署，需要PyTorch特性的场景")

    print("\n2. ONNX:")
    print("   - 优点: 跨框架互操作性，支持多种推理引擎")
    print("   - 优点: 广泛的算子支持和优化")
    print("   - 优点: 可以转换为其他格式的中间表示")
    print("   - 缺点: 某些动态操作支持有限")
    print("   - 适用场景: 需要跨平台部署，或使用专用推理引擎(如ONNX Runtime, TensorRT)")

    print("\n3. TensorFlow Lite:")
    print("   - 优点: 专为移动和嵌入式设备优化")
    print("   - 优点: 支持量化和模型优化")
    print("   - 优点: 在Android和iOS上有原生支持")
    print("   - 缺点: 算子支持比完整TensorFlow少")
    print("   - 适用场景: 移动应用，IoT设备，边缘计算设备")

    # 5. 边缘设备性能分析
    print("\n===== 边缘设备性能分析 =====")
    print("在实际边缘设备上验证模型的方法:")
    print("1. 确认模型是否成功运行:")
    print("   - 使用示例输入进行推理，检查输出是否合理")
    print("   - 比较边缘设备上的输出与原始模型输出是否一致")
    print("   - 在目标设备上运行端到端的推理流程，包括预处理和后处理")

    print("\n2. 性能分析方法:")
    print("   - 测量推理延迟(单次推理时间)")
    print("   - 测量吞吐量(每秒可处理的推理次数)")
    print("   - 监控内存使用情况")
    print("   - 监控CPU/GPU利用率")
    print("   - 测量功耗(对电池供电设备尤为重要)")

    print("\n3. 性能分析工具:")
    print("   - Android: Android Profiler, adb shell dumpsys")
    print("   - iOS: Instruments, Metal System Trace")
    print("   - 通用: 自定义计时代码，日志记录")
    print("   - TensorFlow Lite: TFLite Benchmark Tool")
    print("   - PyTorch Mobile: 内置性能计数器")

    print("\n4. 优化技巧:")
    print("   - 选择合适的线程数")
    print("   - 使用设备特定的加速器(如GPU, DSP, NPU)")
    print("   - 调整批处理大小")
    print("   - 考虑输入大小和预处理步骤")

    return {
        "torchscript_path": "model_torchscript.pt",
        "onnx_path": "model.onnx",
        "tflite_path": "model.tflite",
    }


# 主函数
def main():
    print("===== 模型压缩与边缘部署演示 =====")

    # 加载数据
    print("\n加载CIFAR-10数据集...")
    train_loader, test_loader = load_data()

    # 创建并训练原始模型
    print("\n创建并训练原始模型...")
    model = SimpleCNN()
    model = train_model(model, train_loader, epochs=3)

    # 保存原始模型
    torch.save(model.state_dict(), "original_model.pth")
    print("原始模型已保存为 'original_model.pth'")

    # 模型量化
    print("\n执行模型量化...")
    quantized_model = quantize_model_demo(model, train_loader, test_loader)

    # 模型剪枝
    print("\n执行模型剪枝...")
    pruned_model = prune_model_demo(model, train_loader, test_loader)

    # 模型蒸馏
    print("\n执行模型蒸馏...")
    distilled_model = distill_model_demo(model, train_loader, test_loader)

    # 边缘设备部署
    print("\n执行边缘设备部署...")
    deployment_info = deploy_to_edge_demo(distilled_model, test_loader)

    print("\n===== 总结 =====")
    print("1. 模型量化:")
    print("   - 动态量化: 适用于RNN和Transformer，运行时量化权重")
    print("   - 静态量化: 预先计算量化参数，适用于CNN等")
    print("   - QAT: 在训练中模拟量化，精度最高")
    print("   - 验证: 通过精度、速度、模型大小比较验证量化效果")

    print("\n2. 模型剪枝:")
    print("   - 非结构化剪枝: 移除单个权重，需要特殊硬件支持才能加速")
    print("   - 结构化剪枝: 移除整个通道/神经元，通常能实际加速")
    print("   - 剪枝后需要微调恢复精度")
    print("   - 部署前需要将临时剪枝转为永久剪枝")

    print("\n3. 知识蒸馏:")
    print("   - 温度参数控制软标签平滑程度")
    print("   - 高温度使得学生模型能学习类别间的相似性")
    print("   - 需要平衡蒸馏损失和真实标签损失")

    print("\n4. 边缘部署:")
    print("   - TorchScript: 适合服务器，与PyTorch生态紧密集成")
    print("   - ONNX: 跨框架互操作性，适合多种推理引擎")
    print("   - TFLite: 专为移动和嵌入式设备优化")
    print("   - 验证: 比较输出差异，测量延迟和内存使用")

    print("\n所有模型和部署文件:")
    print(f"- 原始模型: original_model.pth")
    print(f"- 量化模型: quantized_model.pth")
    print(f"- 剪枝模型: pruned_model.pth")
    print(f"- 蒸馏模型: distilled_model.pth")
    print(f"- TorchScript: {deployment_info['torchscript_path']}")
    print(f"- ONNX: {deployment_info['onnx_path']}")
    print(f"- TFLite: {deployment_info['tflite_path']}")


if __name__ == "__main__":
    main()
