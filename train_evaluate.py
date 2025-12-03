import torch
import numpy as np
from sklearn.metrics import f1_score


def train(model, data, optimizer, criterion):
    """单轮训练"""
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    """评估准确率"""
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        accuracy = correct / data.y.size(0)
    return accuracy


def train_with_early_stopping(model, train_data, val_data, optimizer, criterion, patience=50, max_epochs=500):
    """带早停的训练（监控F1，匹配51轮触发）"""
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # 训练
        train_loss = train(model, train_data, optimizer, criterion)

        # 验证
        model.eval()
        with torch.no_grad():
            val_out = model(val_data)
            val_loss = criterion(val_out, val_data.y).item()
            val_pred = val_out.argmax(dim=1)

            val_true = val_data.y.cpu().numpy()
            val_pred_np = val_pred.cpu().numpy()

            val_accuracy = (val_pred_np == val_true).sum() / len(val_true)
            val_f1 = f1_score(val_true, val_pred_np, average='weighted')

        # 每10轮打印
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Accuracy: {val_accuracy * 100:.2f}%, Val F1: {val_f1:.4f}"
            )

        # 早停逻辑（仅更新F1）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # 触发早停
        if patience_counter >= patience:
            print(f"早停触发于第 {epoch + 1} 轮")
            break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"最佳验证集性能 - 损失: {best_val_loss:.4f}, 准确率: {best_val_accuracy * 100:.2f}%")
    return model