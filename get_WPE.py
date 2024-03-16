import numpy as np
import os
from scipy.io import savemat
import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans

# 全局变量
sample_length = 230  # 时间点
n_regions = 116  # AAL116模板 脑区数量

m = 3
tau = 1
ties = 'sequence'
# windox_size = 20
slide_samples = 3

# 将文件夹中的HC/MDD的BOLD矩阵计算得到WPE并拼接
def permEntropy(x, m, tau, ties):
    # 给定模序长度m的所有可能的排列
    possible_perms = np.array([p for p in itertools.permutations(range(m))])
    # 将时间序列分成长度为m的块的方法的数量，在数据中移动tau步
    ncols_partition = len(x) - (m-1)*tau
    # 预先分配一个矩阵，其中每列对应time series x的一个长度为m的块
    partition_mat = np.zeros((m, ncols_partition))
    # 预先分配原始模式矩阵
    rank_mat = np.zeros((m, ncols_partition))

    rows = possible_perms.shape[0]
    # 计算出现模式的频率（第4列）
    pattern_freq = np.append(possible_perms, np.zeros((rows, 1)), axis=1)
    # 计算出现模式的加权频率（第4列）
    weighted_pattern_freq = np.append(possible_perms, np.zeros((rows, 1)), axis=1)
    # 一个矩阵，用于收集时间序列每个分区片段的均值(第1行)和权重(第2行)
    pattern_mat = np.zeros((2, ncols_partition))

    # 将时间序列分割成长度为m的块，每次推进tau步
    for i in range(ncols_partition):
        partition_mat[:,i] = x[i:i+m]
        unique_elements, idx_tmp, counts = np.unique(partition_mat[:,i], return_index=True, return_counts=True)
        sequence_rank = np.searchsorted(unique_elements, partition_mat[:,i])
        if sequence_rank.shape[0] != unique_elements.shape[0]:
            non_unique = unique_elements[np.where(counts>1)[0]]
            idx_non_unique = np.isin(partition_mat[:,i], non_unique)    # 索引分区片段中的非唯一项
            # 使用哪一种方法来调节ties?
            if ties == 'GaussianNoise':
                partition_mat[idx_non_unique,i] = partition_mat[idx_non_unique,i]+np.random.random(len(partition_mat[idx_non_unique,i]))*(10**(-10))    # 向非唯一项添加最小的高斯噪声
                unique_elements = np.unique(partition_mat[:,i])
                sequence_rank = np.searchsorted(unique_elements, partition_mat[:,i])

            # 按代码片段中出现的顺序解析ties
            elif ties == 'sequence':
                ties_sorted = sorted(partition_mat[:,i])
                seq_rank_tmp = np.searchsorted(ties_sorted, partition_mat[:,i])
                seq_rank_tmp[idx_non_unique] = -1
                possible_ranks = np.array(range(seq_rank_tmp.shape[0])) # possible ranks in the motif
                missing_ranks = np.setdiff1d(possible_ranks, seq_rank_tmp)  # those ranks not assigned (the former ties)
                seq_rank_tmp[idx_non_unique] = missing_ranks
                sequence_rank = seq_rank_tmp

        rank_mat[:,i] = sequence_rank   # keep in rank sequence in rank matrix
        pattern_idx = np.where((possible_perms == rank_mat[:,i]).all(axis=1))[0]    # 检查第i个分区片段等同于possible_perms中的哪个
        pattern_freq[pattern_idx, m] += 1 # 计算每个模式出现的次数
        pattern_mat[0,i] = np.mean(partition_mat[:,i])
        pattern_mat[1,i] = (1/m)*sum((partition_mat[:,i]-pattern_mat[0,i])**2)
        weighting_factor_i = pattern_mat[1,i]
        weighted_pattern_freq[pattern_idx, m] = weighted_pattern_freq[pattern_idx, m] + 1*weighting_factor_i

    # 计算x上的排列熵 permutation entropy
    prob_vec = pattern_freq[:,m]/ncols_partition    # 每个模式出现的(未加权的)概率向量
    PE = -np.sum(prob_vec[prob_vec>0]*np.log2(prob_vec[prob_vec>0]))   # 排列熵是所有模式概率(pi) *该概率的对数 的负和;丢弃零项(根据定义应为零)
    PE_norm = PE / np.log2(possible_perms.shape[0]) # 排列熵归一化到区间[0 1];分母等于log2(阶乘(m))

    # 计算x上的加权排列熵 weighted permutation entropy
    weighted_rank_patterns = weighted_pattern_freq[weighted_pattern_freq[:,m]>0, m]
    weighted_prob_vec = weighted_rank_patterns/np.sum(weighted_rank_patterns)
    WPE = -np.sum(weighted_prob_vec*np.log2(weighted_prob_vec))
    WPE_norm = WPE / np.log2(possible_perms.shape[0])

    return PE, PE_norm, WPE, WPE_norm

# 计算单个subject的复杂度矩阵
def get_single_WPE(k, idx_vec, X, nr_windows, WPE_mat, window_size):
    WPE_timeseries = np.zeros((nr_windows, n_regions))
    for j in range(n_regions):
        x = X[:, j]

        PE_timeseries = []
        for i in range(nr_windows):
            current_window = x[idx_vec[i]: idx_vec[i] + window_size]
            _, _, _, WPE_norm_i = permEntropy(current_window, m, tau, ties)
            PE_timeseries.append(WPE_norm_i)

        WPE_timeseries[:, j] = PE_timeseries

    if k == 0:
        WPE_mat = WPE_timeseries
    else:
        WPE_mat = np.append(WPE_mat, WPE_timeseries, axis=0)
    return WPE_mat


# 私有数据 MDD168/HC89
def get_private_WPE_mat(window_size):
    idx_vec = list(range(0, sample_length - window_size, slide_samples))
    nr_windows = len(idx_vec)

    # directory = r'E:\fmri\python分类\test2\HC'
    directory = r'E:\fmri\python分类\test2\MDD'
    namelist = os.listdir(directory)

    WPE_mat = np.zeros((nr_windows, n_regions))
    for k in range(len(namelist)):
        filename = namelist[k]
        file = os.path.join(directory, filename)
        X = np.loadtxt(file)
        if X.shape[0] != sample_length:
            print(filename)
            continue
        WPE_mat = get_single_WPE(k, idx_vec, X, nr_windows, WPE_mat, window_size)

    savefile = 'D:\Desktop\学校\实验室\cluster plot\WPE_timeseries_MDD168_py.mat'
    # savefile = 'D:\Desktop\学校\实验室\cluster plot\WPE_timeseries_HC89_py.mat'
    # WPE_mat = np.array(WPE_mat, dtype=np.object)
    savemat(savefile, {'WPE_mat':WPE_mat})

# REST-meta-MDD
def get_REST_WPE_mat(window_size):
    file_dir = r'E:\REST-meta-MDD-proc\mdd_fmri_proc\fmri_list_aal116.pt'
    label_dir = r'E:\REST-meta-MDD-proc\mdd_fmri_proc\label_aal116.npy'
    files = torch.load(file_dir)
    labels = np.load(label_dir)
    idx_vec = list(range(0, sample_length - window_size, slide_samples))
    nr_windows = len(idx_vec)
    WPE_mat_MDD = np.zeros((nr_windows, n_regions))
    WPE_mat_HC = np.zeros((nr_windows, n_regions))
    i = 0
    j = 0
    for k in range(len(files)):
        X = files[k]
        if X.shape[0] != sample_length:
            continue
        lb = labels[k]
        if lb:  # MDD正类，HC负类
            WPE_mat_MDD = get_single_WPE(i,idx_vec,X,nr_windows,WPE_mat_MDD)
            i += 1
        else:
            WPE_mat_HC = get_single_WPE(j,idx_vec,X,nr_windows,WPE_mat_HC)
            j += 1
    print(f"len(fmri_MDD): {i}")
    print(f"len(fmri_HC): {j}")
    savefile = r'D:\Desktop\学校\实验室\cluster plot\REST_META_MDD.mat'
    savemat(savefile, {'WPE_mat_MDD': WPE_mat_MDD})
    savefile = r'D:\Desktop\学校\实验室\cluster plot\REST_META_HC.mat'
    savemat(savefile, {'WPE_mat_HC': WPE_mat_HC})


# kmeans cluster, 得到状态序列
def get_state_series(k):
    ch = input('1. HC\t2.MDD\nplease input:\t')
    if ch == '1':
        savepath = r'D:\Desktop\学校\实验室\cluster plot\REST-meta-HC'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        file = r'D:\Desktop\学校\实验室\cluster plot\REST_META_HC.mat'
        WPE_mat = sio.loadmat(file)['WPE_mat_HC']
        print(f"len(fmri_HC): {len(WPE_mat)}")
    else:
        savepath = r'D:\Desktop\学校\实验室\cluster plot\REST-meta-MDD'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        file = r'D:\Desktop\学校\实验室\cluster plot\REST_META_MDD.mat'
        WPE_mat = sio.loadmat(file)['WPE_mat_MDD']
        print(f"len(fmri_MDD): {len(WPE_mat)}")

    # file = r'E:\fmri\test\hc_all\WPE_timeseries_newHC.mat'
    # WPE_mat = sio.loadmat(file)['WPE_mat']

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=20)
    kmeans.fit(WPE_mat)

    # 获取聚类结果
    idx = kmeans.labels_
    centroids = kmeans.cluster_centers_
    sumd = kmeans.inertia_

    print('Cluster labels shape:', idx.shape)
    print('Cluster centroids shape:', centroids.shape)
    # print('Sum of squared distances:', sumd)

    [n_node, regions] = WPE_mat.shape
    print('n_node:', n_node, '\nregions:', regions)

    n_window = 70

    n_participant = n_node // n_window

    occupancy = np.zeros((n_participant, k))
    state = np.zeros((n_participant, k))
    for i in range(n_participant):
        first = n_window * i
        last = n_window * (i + 1)
        for j in range(first, last):
            s = idx[j]
            state[i][s] += 1
        for j in range(k):
            occupancy[i][j] = state[i][j] / n_window

    node_state = [[] for i in range(k)]
    # print(node_state)   # [[], [], [], []]
    for i in range(n_node):
        WPE = sum(WPE_mat[i, :]) / regions
        node_state[idx[i]].append(WPE)

    med_state = np.zeros(k)
    for i in range(k):
        med_state[i] = np.median(node_state[i])

    # 只是聚类分出来四个状态，但是没有顺序大小之分，现在要让0对应的WPE最小，3对应的最大
    # 即根据中位数来比较大小 排序后按照该顺序对与所有与状态相关的变量进行变换
    sorted_indices = np.argsort(med_state)
    print(sorted(med_state))

    # 使状态按sorted_indices规定的顺序重排
    occupancy = np.take(occupancy, sorted_indices, axis=1)
    state = np.take(state, sorted_indices, axis=1)
    centroids = np.take(centroids, sorted_indices, axis=0)
    for i in range(len(idx)):
        idx[i] = np.where(sorted_indices == idx[i])[0][0]
    node_state = [node_state[i] for i in sorted_indices]

    plt.figure()
    for i in range(n_participant):
        plt.scatter(occupancy[i, :], range(k), s=3)

    plt.ylabel('State')
    plt.xlabel('Propotion of time')
    plt.ylim(-1, k)
    plt.xlim(-0.05, 1.05)
    plt.yticks(range(k))
    plt.title('Occupancy')
    plt.savefig(os.path.join(savepath, 'Occupancy.png'))
    # plt.show()

    plt.figure()
    for i in range(k):
        array = node_state[i]
        plt.scatter(array, [i] * len(array), s=3)

    plt.ylabel('State')
    plt.xlabel('WPE')
    plt.ylim(-1, k)
    plt.xlim(-0.05, 1.05)
    plt.yticks(range(k))
    plt.title('Complexity')
    plt.savefig(os.path.join(savepath, 'Complexity.png'))
    # plt.show()

    # 每个participant的各状态的中位数
    med = np.zeros((n_participant, k))
    for i in range(n_participant):
        first = n_window * i
        last = n_window * (i + 1)
        WPE_state = [[] for i in range(k)]
        for j in range(first, last):
            avg = np.sum(WPE_mat[j, :]) / regions
            WPE_state[idx[j]].append(avg)
        for j in range(k):
            med[i, j] = np.median(WPE_state[j])

    # 保存变量
    idx = np.array(idx)
    idx = idx.reshape(n_participant, n_window)  # n_participant × n_window
    # np.save(os.path.join(savepath, 'idx.npy'), idx)
    np.savetxt(os.path.join(savepath, 'idx.txt'), idx)

    savename = os.path.join(savepath, r'data.xlsx')
    writer = pd.ExcelWriter(savename)  # 生成一个excel文件
    med = pd.DataFrame(med)  # 利用pandas库对数据进行格式转换
    med.to_excel(writer, 'WPE')  # 数据写入excel文件
    occupancy = pd.DataFrame(occupancy)
    occupancy.to_excel(writer, 'Occupancy')
    state = pd.DataFrame(state)
    state.to_excel(writer, 'Propotion of time')
    writer.close()  # 保存excel文件

if __name__ == '__main__':
    # window_size = 20
    # # get_private_WPE_mat(window_size=window_size)
    # get_REST_WPE_mat(window_size)
    k = 4
    get_state_series(k=k)

