import torch
from sklearn.neighbors import NearestNeighbors


def construct_graph(X_train, method='sklearn_knn', k=5):
    """构建KNN图（k=5，生成双向边）"""
    if method == 'sklearn_knn':
        return construct_graph_sklearn_knn(X_train, k)
    else:
        raise ValueError(f"不支持的图构建方法: {method}")


def construct_graph_sklearn_knn(X_train, k=5):
    """SKlearn KNN构建图（欧式距离，k=5）"""
    num_nodes = len(X_train)
    k = min(k, num_nodes - 1)

    # KNN计算
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)

    rows = []
    cols = []

    # 生成双向边
    for i in range(num_nodes):
        neighbors = indices[i]
        neighbors = neighbors[neighbors != i]  # 排除自身
        neighbors = neighbors[:k]

        for j in neighbors:
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index

