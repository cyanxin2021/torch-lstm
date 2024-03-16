import numpy as np
import pandas as pd
import os
from scipy.io import loadmat


# 获取有效的样本id
def get_effective_id():
    file = 'subjects.xlsx'
    label_df = pd.read_excel(file)
    id_temp = list(label_df['id'].astype(str))
    id_temp = [id.rjust(3, '0') for id in id_temp]
    label_temp = list(label_df['target'])
    # print(id_temp)
    # print(label_temp)
    label_dict = dict(zip(id_temp, label_temp))
    # 使用 pop 移除无效的键值对
    new_dict = {key: val for key, val in label_dict.items() if
                (key != '112' and key != '1080-039' and key != '1080-045')}
    # mdd_id = []
    # hc_id = []
    # for key in new_dict.keys():
    #     if new_dict[key] == 'MDD':
    #         mdd_id.append(key)
    #     else:
    #         hc_id.append(key)
    # # 将list保存为txt
    # f = open("mdd_id.txt", 'w')
    # f.write(str(mdd_id))
    # f.close()
    # f = open("hc_id.txt", 'w')
    # f.write(str(hc_id))
    # f.close()
    # # 将txt文本读取为list
    # file = open(path, 'r')
    # rdlist = eval(file.read())
    # file.close()
    return new_dict


# 状态序列数据
def get_bold(file1, file2):
    sub_mdd = np.loadtxt(file1)
    sub_hc = np.loadtxt(file2)
    label = []

    for i in sub_hc:
        label.append(0)
    for i in sub_mdd:
        label.append(1)

    state_mrtrix = np.append(sub_hc, sub_mdd, axis=0)
    return state_mrtrix, label


# 静态FC数据:f1——mdd，f2——hc
def get_fc(path1, path2):
    label_dict = get_effective_id()

    final_fc = []
    final_label = []
    # ch = 1  # 1是MDD，0是HC
    #
    # for id in label_dict.keys():
    #     file = 'z' + id + '.txt'
    #     if label_dict[id] == 'MDD':
    #         ch = 1
    #         path = os.path.join(path1, file)
    #         if not os.path.exists(path):
    #             print("MDD-" + id + " don't have fmri data!")
    #             continue
    #     else:
    #         ch = 0
    #         path = os.path.join(path2, file)
    #         if not os.path.exists(path):
    #             print("HC-" + id + " don't have fmri data!")
    #             continue
    #     fc = np.loadtxt(path)  # 对称矩阵
    #     fc_temp = []
    #     for i in range(len(fc)):
    #         for j in range(i + 1, len(fc)):
    #             fc_temp.append(fc[i][j])
    #     final_fc.append(fc_temp)
    #     final_label.append(ch)

    mdd_list = os.listdir(path1)
    hc_list = os.listdir(path2)
    for file in hc_list:
        filename = file.split('.')[0][1:]
        if filename not in label_dict.keys():
            # print("HC-" + filename + " don't have fmri data!")
            continue
        path = os.path.join(path2, file)
        fc = np.loadtxt(path)  # 对称矩阵
        fc_temp = []
        for i in range(len(fc)):
            for j in range(i + 1, len(fc)):
                fc_temp.append(fc[i][j])
        final_fc.append(fc_temp)
        final_label.append(0)
    for file in mdd_list:
        filename = file.split('.')[0][1:]
        if filename not in label_dict.keys():
            # print("MDD-" + filename + " don't have fmri data!")
            continue
        path = os.path.join(path1, file)
        fc = np.loadtxt(path)  # 对称矩阵
        fc_temp = []
        for i in range(len(fc)):
            for j in range(i + 1, len(fc)):
                fc_temp.append(fc[i][j])
        final_fc.append(fc_temp)
        final_label.append(1)

    return np.array(final_fc), final_label


# WPE
def get_WPE(file1, file2):
    # file1=MDD,file2=HC
    mdd = loadmat(file1, mat_dtype=True)['WPE_mat']
    hc = loadmat(file2, mat_dtype=True)['WPE_mat']
    # print(len(mdd[0]))
    # print(len(hc[0]))
    final_WPE = []
    final_label = []  # 1是MDD，0是HC
    window_size = 70
    wpe_temp = []
    for i in range(len(hc)):
        for j in range(len(hc[0])):
            wpe_temp.append(hc[i][j])
        if (i % window_size) == 69:
            final_WPE.append(wpe_temp)
            wpe_temp = []
            final_label.append(0)
    for i in range(len(mdd)):
        for j in range(len(mdd[0])):
            wpe_temp.append(mdd[i][j])
        if (i % window_size) == 69:
            final_WPE.append(wpe_temp)
            wpe_temp = []
            final_label.append(1)
    return np.array(final_WPE), final_label

# label_dict = get_effective_id()
# num1 = 0
# num2 = 0
# for key in label_dict.keys():
#     if label_dict[key] == 'MDD':
#         num1 += 1
#     else:
#         num2 += 1
# print(num1, num2)
