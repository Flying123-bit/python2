import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings


def load_data(sample_fraction=0.2, random_state=42):
    """加载NSL-KDD数据集（20%分层抽样）"""
    # 固定文件路径
    train_file = r'D:\code\pythonProject2\NSL-KDD-DataSet-master\KDDTrain+.txt'
    test_file = r'D:\code\pythonProject2\NSL-KDD-DataSet-master\KDDTest+.txt'

    # 检查文件
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")

    print(f"训练文件路径: {train_file}")
    print(f"测试文件路径: {test_file}")

    # 加载原始数据（43列）
    train_data = pd.read_csv(train_file, header=None)
    test_data = pd.read_csv(test_file, header=None)

    # 分层抽样（核心：按标签抽样，保留小类）
    if sample_fraction is not None and sample_fraction < 1.0:
        print(f"正在抽样 {sample_fraction * 100}% 的数据...")

        train_labels = train_data.iloc[:, -1]
        train_data_sampled = train_data.groupby(train_labels, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=random_state)
        )

        test_labels = test_data.iloc[:, -1]
        test_data_sampled = test_data.groupby(test_labels, group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=random_state)
        )

        train_data = train_data_sampled.reset_index(drop=True)
        test_data = test_data_sampled.reset_index(drop=True)

    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    """数据预处理（生成137维特征）"""
    print(f"训练数据集形状: {train_data.shape}")
    print(f"测试数据集形状: {test_data.shape}")

    num_columns = len(train_data.columns)
    print(f"数据集列数：{num_columns}")

    # 43列命名（匹配NSL-KDD格式）
    base_columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
        'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
        'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'extra_col_0', 'Label'
    ]
    columns = base_columns[:num_columns]
    print(f"使用列名数量: {len(columns)}")

    train_data.columns = columns
    test_data.columns = columns

    # 分离特征和标签
    y_train = train_data['Label']
    y_test = test_data['Label']

    # 标签编码（22类）
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)

    print(f"标签编码映射: {dict(zip(le.classes_, range(len(le.classes_))))}")

    # 特征列
    feature_columns = columns[:-1]
    X_train = train_data[feature_columns].copy()
    X_test = test_data[feature_columns].copy()

    print("数据预处理步骤:")

    # 分类特征识别
    categorical_columns = ['protocol_type', 'service', 'flag', 'extra_col_0']
    numeric_columns = [col for col in X_train.columns if col not in categorical_columns]
    print(f"分类特征: {categorical_columns}")
    print(f"数值特征: {len(numeric_columns)}个")

    # 分类特征OneHot编码（生成137维特征）
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            train_categorical_encoded = encoder.fit_transform(X_train[categorical_columns])
            test_categorical_encoded = encoder.transform(X_test[categorical_columns])

        categorical_feature_names = encoder.get_feature_names_out(categorical_columns)

        train_categorical_df = pd.DataFrame(train_categorical_encoded,
                                            columns=categorical_feature_names,
                                            index=X_train.index)
        test_categorical_df = pd.DataFrame(test_categorical_encoded,
                                           columns=categorical_feature_names,
                                           index=X_test.index)

        X_train_processed = X_train.drop(columns=categorical_columns)
        X_test_processed = X_test.drop(columns=categorical_columns)

        X_train_processed = pd.concat([X_train_processed, train_categorical_df], axis=1)
        X_test_processed = pd.concat([X_test_processed, test_categorical_df], axis=1)
    else:
        X_train_processed = X_train.copy()
        X_test_processed = X_test.copy()

    print(f"编码后特征数量: {X_train_processed.shape[1]}")

    # 数据清洗
    print("处理NaN和inf值...")
    X_train_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_processed.replace([np.inf, -np.inf], np.nan, inplace=True)

    train_nan_count = X_train_processed.isna().sum().sum()
    test_nan_count = X_test_processed.isna().sum().sum()
    print(f"训练集NaN值数量: {train_nan_count}")
    print(f"测试集NaN值数量: {test_nan_count}")

    if train_nan_count > 0:
        X_train_processed = X_train_processed.fillna(X_train_processed.mean())
    if test_nan_count > 0:
        X_test_processed = X_test_processed.fillna(X_test_processed.mean())

    # 标准化
    print("标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    print(f"最终训练集形状: {X_train_scaled.shape}")
    print(f"最终测试集形状: {X_test_scaled.shape}")

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, le