import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import math
from sklearn.utils import resample

# 文件夹路径
mdd_folder = './data/SI'
nc_folder = './data/NSI'


# 加载数据
def load_data(folder):
    """
    将类别对应的所有样本数据转化成一维
    :return: 样本的行列标签以及每个样本的一维数据
    """
    data = []
    indices = None
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, index_col=0)
            if indices is None:
                # 将行列标签转换成列表
                indices = df.index.tolist()
                columns = df.columns.tolist()
            matrix = df.values.flatten() # 数据一维化，每个样本的所有数据化成一维数据
            data.append(matrix)
    return np.array(data), indices, columns


mdd_data, row_indices, col_indices = load_data(mdd_folder)
nc_data, _, _ = load_data(nc_folder)

# 打印数据加载后的形状
print("MDD data shape:", mdd_data.shape)
print("NC data shape:", nc_data.shape)

# 计算Kolmogorov-Smirnov统计量
num_features = mdd_data.shape[1] # 特征数量
num_bootstrap_samples = 500  # 设置bootstrap样本数量
ks_statistics_bootstrap = np.zeros((num_bootstrap_samples, num_features))
ks_p_values_bootstrap = np.zeros((num_bootstrap_samples, num_features))

for b in range(num_bootstrap_samples):
    # Bootstrap抽样，每次抽样的样本大小与原始数据集相同，有放回
    mdd_resampled = resample(mdd_data, n_samples=mdd_data.shape[0], replace=True)
    nc_resampled = resample(nc_data, n_samples=nc_data.shape[0], replace=True)

    for i in range(num_features):
        mdd_feature = mdd_resampled[:, i]
        nc_feature = nc_resampled[:, i]
        ks_stat, p_value = ks_2samp(mdd_feature, nc_feature)
        ks_statistics_bootstrap[b, i] = ks_stat
        ks_p_values_bootstrap[b, i] = p_value

# 计算平均Kolmogorov-Smirnov统计量和p值
ks_statistics_mean = np.mean(ks_statistics_bootstrap, axis=0)
ks_p_values_mean = np.mean(ks_p_values_bootstrap, axis=0)

# 保留前500个特征
top_n_features = 500
top_features_indices = np.argsort(ks_statistics_mean)[-top_n_features:]

# 筛选后的特征
mdd_selected_features = mdd_data[:, top_features_indices]
nc_selected_features = nc_data[:, top_features_indices]

# 使用（行索引，列索引）作为列名
selected_features_indices = [(i // len(col_indices), i % len(col_indices)) for i in top_features_indices]
selected_feature_names = [(row_indices[row], col_indices[col]) for row, col in selected_features_indices]

# 保存筛选后的特征
mdd_selected_df = pd.DataFrame(mdd_selected_features, columns=selected_feature_names)
nc_selected_df = pd.DataFrame(nc_selected_features, columns=selected_feature_names)

mdd_selected_df.to_csv('D:/All result/si vs nsi/10fold/gut/SI_selected_features.csv', index=False)
nc_selected_df.to_csv('D:/All result/si vs nsi/10fold/gut/NSI_selected_features.csv', index=False)

# 保存筛选特征的KS统计量和p值
selected_ks_statistics = ks_statistics_mean[top_features_indices]
selected_p_values = ks_p_values_mean[top_features_indices]

ks_p_df = pd.DataFrame({
    'Feature': selected_feature_names,
    'KS_Statistic': selected_ks_statistics,
    'p_value': selected_p_values
})

ks_p_df.to_csv('D:/All result/si vs nsi/10fold/gut/selected_features_ks_p_values.csv', index=False)