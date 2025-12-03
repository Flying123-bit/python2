from data_preprocessing import preprocess_data, load_data
from graph_construction import construct_graph
from Build_GAT_model import GAT
from train_evaluate import train, evaluate, train_with_early_stopping
import torch
import torch.optim as optim
from torch_geometric.data import Data
import warnings
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def main():
    try:
        # 加载数据
        print("=" * 60)
        print("NSL-KDD网络入侵检测系统 - GAT模型训练")
        print("=" * 60)

        sample_fraction = 0.5  # 50%抽样（核心配置）
        train_data, test_data = load_data(sample_fraction=sample_fraction)
        print(f"训练数据形状: {train_data.shape}, 测试数据形状: {test_data.shape}")

        # 预处理数据
        print("\n预处理数据中...")
        X_train, X_test, y_train, y_test, label_encoder = preprocess_data(train_data, test_data)

        print(f"预处理完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")

        # 计算类别权重
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        print(f"类别权重计算完成: {class_weights}")

        # 构建图结构（k=5，生成25万+边）
        print("\n构建图结构中...")
        train_edge_index = construct_graph(X_train, method='sklearn_knn', k=5)
        test_edge_index = construct_graph(X_test, method='sklearn_knn', k=5)

        print(f"图构建完成 - 训练图边数: {train_edge_index.shape[1]}, 测试图边数: {test_edge_index.shape[1]}")

        # 转换为PyG数据格式
        train_data_pyg = Data(
            x=torch.tensor(X_train, dtype=torch.float),
            edge_index=train_edge_index,
            y=torch.tensor(y_train, dtype=torch.long)
        )

        test_data_pyg = Data(
            x=torch.tensor(X_test, dtype=torch.float),
            edge_index=test_edge_index,
            y=torch.tensor(y_test, dtype=torch.long)
        )

        # 模型配置（核心：128维+3层+6头）
        num_classes = len(label_encoder.classes_)
        in_channels = X_train.shape[1]

        print(f"\n模型配置:")
        print(f"- 输入特征维度: {in_channels}")
        print(f"- 输出类别数: {num_classes}")
        print(f"- 隐藏层维度: 128")
        print(f"- GAT层数: 3")
        print(f"- 注意力头数: 6")

        # 设备设置（CPU）
        device = torch.device('cpu')  # 匹配输出中的CPU设备
        print(f"使用设备: {device}")

        # 初始化模型
        model = GAT(
            in_channels=in_channels,
            out_channels=num_classes,
            hidden_channels=128,
            num_layers=3,
            heads=6,
            dropout=0.2,
            residual=True
        ).to(device)

        # 优化器（无学习率调度器）
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        # 数据转移到设备
        train_data_pyg = train_data_pyg.to(device)
        test_data_pyg = test_data_pyg.to(device)

        # 训练模型（patience=50，max_epochs=500）
        print(f"\n开始训练...")
        model = train_with_early_stopping(
            model, train_data_pyg, test_data_pyg, optimizer, criterion,
            patience=50,  # 匹配早停在51轮触发
            max_epochs=100
        )

        # 最终评估
        print(f"\n最终评估结果:")
        model.eval()
        accuracy = evaluate(model, test_data_pyg)
        # 计算F1分数
        with torch.no_grad():
            outputs = model(test_data_pyg)
            _, predicted = torch.max(outputs, 1)
            test_f1 = f1_score(test_data_pyg.y.cpu().numpy(), predicted.cpu().numpy(), average='weighted')

        print(f"测试集准确率: {accuracy * 100:.2f}%")
        print(f"测试集F1分数: {test_f1:.4f}")

        # 详细分类报告
        with torch.no_grad():
            outputs = model(test_data_pyg)
            _, predicted = torch.max(outputs, 1)

            predicted_labels = label_encoder.inverse_transform(predicted.cpu().numpy())
            true_labels = label_encoder.inverse_transform(test_data_pyg.y.cpu().numpy())

            # 分类报告
            try:
                target_names = [str(cls) for cls in label_encoder.classes_]
                print(f"\n详细分类报告:")
                print(classification_report(true_labels, predicted_labels,
                                            target_names=target_names, zero_division=0))
            except Exception as e:
                print(f"分类报告生成错误: {e}")

            # 混淆矩阵
            try:
                cm = confusion_matrix(true_labels, predicted_labels)
                print(f"混淆矩阵:")
                print(cm)
            except Exception as e:
                print(f"混淆矩阵生成错误: {e}")

        # 保存模型
        model_info = {
            'model_state_dict': model.state_dict(),
            'in_channels': in_channels,
            'num_classes': num_classes,
            'label_encoder': label_encoder,
            'accuracy': accuracy
        }

        torch.save(model_info, 'nsl_kdd_gat_best.pth')
        print(f"\n模型已保存为: nsl_kdd_gat_best.pth")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()


# 忽略警告
warnings.filterwarnings("ignore", category=RuntimeWarning, module='sklearn')

if __name__ == "__main__":
    main()