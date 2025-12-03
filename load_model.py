import torch
from Build_GAT_model import GAT

# 1. 初始化模型（必须和训练时的层结构、参数名完全一致）
model = GAT(
    in_channels=140,  # 确认训练时的参数名是in_channels而非in_features
    out_channels=22,
    hidden_channels=128,
    num_layers=3,  # 确认训练时是3层GAT，对应convs.0/1/2
    heads=6
)

# 2. 加载保存的字典，提取模型权重（关键修正）
try:
    # 先加载完整字典（关闭weights_only，适配LabelEncoder）
    checkpoint = torch.load(
        "nsl_kdd_gat_best.pth",
        map_location="cpu",
        weights_only=False
    )
    # 只提取模型权重（model_state_dict）加载
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型加载成功！")

    # 可选：提取其他保存的信息
    print(f"训练时的最佳准确率：{checkpoint['accuracy']}")
    print(f"模型输入维度：{checkpoint['in_channels']}")

except FileNotFoundError:
    print("错误：未找到nsl_kdd_gat_best.pth文件，请检查路径！")
except KeyError as e:
    print(f"字典键缺失：{e}，请确认训练时保存了'model_state_dict'键！")
except RuntimeError as e:
    print(f"模型结构不匹配：{e}，请检查GAT类的层定义！")