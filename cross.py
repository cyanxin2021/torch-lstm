import numpy as np
import pandas as pd
import datetime
import os

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


import get_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score, classification_report, precision_score, recall_score

from imblearn.under_sampling import RandomUnderSampler



def filename():
    current_datetime = datetime.datetime.now()
    formatted_datatime = current_datetime.strftime("%Y-%m-%d-%H-%M")

    file_name_prefix = "file"
    file_extension = ".csv"

    file_name = f"{file_name_prefix}_{formatted_datatime}{file_extension}"
    return file_name

def bold_train():   # (257, 70) (257,)
    # BOLD的文件
    # file1 = 'MDD-89.txt'
    # file1 = r'F:\fmri\python分类\test2\MDD-168.txt'
    # file2 = r'F:\fmri\python分类\test2\HC-89.txt'
    file1 = r'D:\Desktop\学校\实验室\cluster plot\REST-meta-MDD\idx.txt'
    file2 = r'D:\Desktop\学校\实验室\cluster plot\REST-meta-HC\idx.txt'
    train, label = get_data.get_bold(file1, file2)
    return train, label

def fc_train(): # (257, 6670) (257,)
    # FC的文件
    path1 = 'F:/fmri/MDD169vHC91/MDD-fmri-aal116'
    path2 = 'F:/fmri/MDD169vHC91/HC-fmri-aal116'
    train, label = get_data.get_fc(path1, path2)
    return train, label


def wpe_train():    # (257, 8120) (257,)
    # 复杂度矩阵
    file1 = r'F:\fmri\test\MDD_all\WPE_timeseries_randomMDD.mat'
    file2 = r'F:\fmri\test\hc_all\WPE_timeseries_newHC.mat'
    train, label = get_data.get_WPE(file1, file2)
    return train, label


def fc_wpe():
    # FC和WPE结合进行分类
    train1,label1 = fc_train()
    train2,label2 = wpe_train()
    train = np.append(train1, train2, axis=1)   # 横向拼接
    return train, label1


def bold_fc():
    # 状态序列和FC结合进行分类
    train1, label = bold_train()
    train2, _ = fc_train()
    train = np.append(train1, train2, axis=1)   # 横向拼接
    return train, label


def bold_wpe():
    # 状态序列和WPE结合进行分类
    train1, label = bold_train()
    train2, _ = wpe_train()
    train = np.append(train1, train2, axis=1)   # 横向拼接
    return train, label


def bold_wpe_fc():
    # 状态序列、WPE、FC结合进行分类
    train1, label = bold_train()
    train2, _ = fc_wpe()
    train = np.append(train1, train2, axis=1)   # 横向拼接
    return train, label


if __name__ == '__main__':
    train, label = bold_train()
    para = 'Optimize'   # 'none' vs 'Optimize'
    mymodel = 'SVM'  # 'SVM' vs 'DT'
    index = 'BOLD-' + mymodel
    sample = '否,MDD:HC=169:89'
    # file = './result/' + mymodel + '.csv'
    file = './result/aaa.csv'

    ch = input("1. 状态序列\t2. 静态FC\t3. WPE矩阵\t4. 静态FC+WPE矩阵\t5. 状态序列+静态FC\t6. 状态序列+WPE矩阵\t7. 状态序列+WPE矩阵+静态FC\n")
    if ch == '1':
        # BOLD
        train, label = bold_train()
        index = 'BOLD-' + mymodel
    elif ch == '2':
        # FC的文件
        train, label = fc_train()
        index = 'FC-' + mymodel
    elif ch == '3':
        # WPE的文件
        train, label = wpe_train()
        index = 'WPE-' + mymodel
    elif ch == '4':
        # FC + WPE
        train, label = fc_wpe()
        index = 'FC+WPE-' + mymodel
    elif ch == '5':
        # BOLD + FC
        train, label = bold_fc()
        index = 'BOLD+FC-' + mymodel
    elif ch == '6':
        # BOLD + WPE
        train, label = bold_wpe()
        index = 'BOLD+WPE-' + mymodel
    elif ch == '7':
        # BOLD + WPE
        train, label = bold_wpe_fc()
        index = 'BOLD+WPE+FC-' + mymodel



    ch_ = input("\n是否需要欠采样: y or n\n")
    if ch_ == 'y':
        rus = RandomUnderSampler(random_state=300)
        train, label = rus.fit_resample(train, label)
        sample = '欠采样,MDD:HC=89:89'

    balanced_res = []
    accuracy_res = []
    sensitivity_res = []
    specificity_res = []
    auc = []
    f1 = []
    precision_res = []  # 查准率
    recall_res = [] # 查全率
    all_tp = []
    all_tn = []
    all_fp = []
    all_fn = []

    skf = StratifiedKFold(n_splits=10, shuffle=True,random_state=500)
    fold = 0
    for train_index,test_index in skf.split(train, label):
        print('#####fold ', fold)
        X_train, X_test = train[train_index], train[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]

        # 除状态序列以外的都作归一化处理
        if ch != '1':
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


        # 模型选择SVM.SVC, 默认参数
        # model = SVC(random_state=2)
        # model = DecisionTreeClassifier(random_state=2)
        model = SVC(random_state=2,C=5,kernel='rbf',probability=True,tol=0.001)
        # model = DecisionTreeClassifier(random_state=2,max_depth=10,min_samples_leaf=5,min_samples_split=50)


        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        # print(classification_report(Y_test, y_pred))
        # print("\n\n")

        result = model.score(X_test, Y_test)
        accuracy_res.append(result)
        balanced_res.append(balanced_accuracy_score(Y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(Y_test, y_pred).ravel()   # 让多维数组变成一维数组
        # print('tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp)
        all_tp.append(tp)
        all_tn.append(tn)
        all_fp.append(fp)
        all_fn.append(fn)
        sensitivity_res.append(tp/(tp+fn))  # 真阳性率
        specificity_res.append(tn/(tn+fp))    # 真阴性率
        auc.append(roc_auc_score(Y_test,y_pred))
        f1.append(f1_score(Y_test,y_pred,zero_division=1))  # f1_score函数的zero_division参数用于控制当分母为0时的行为。当该参数为0时，分母为0时返回0；当该参数为1时，分母为0时返回1；当该参数为其他数值时，分母为0时返回该数值。该参数的默认值为’warn’，表示当分母为0时会发出警告并返回0。当zero_division参数为1时，无论分母是否为0，F1分数都为1，即预测结果完全正确
        precision_res.append(precision_score(Y_test, y_pred))
        recall_res.append(recall_score(Y_test, y_pred))
        epoch = [(para, accuracy_res[fold], balanced_res[fold], sensitivity_res[fold], specificity_res[fold],
                  auc[fold], f1[fold], precision_res[fold], recall_res[fold], tp, tn, fp, fn, sample)]
        analyse_table = pd.DataFrame(epoch, index=[fold])
        analyse_table.to_csv(file, mode='a', header=False)

        fold += 1

    final_result = [np.mean(accuracy_res), np.mean(balanced_res), np.mean(sensitivity_res),
                    np.mean(specificity_res), np.mean(auc), np.mean(f1), np.mean(precision_res),
                    np.mean(recall_res), np.mean(all_tp), np.mean(all_tn), np.mean(all_fp), np.mean(all_fn)]
    mid = tuple([para]) + tuple((round(i, 4) for i in final_result[:8])) + tuple(final_result[8:]) + tuple([sample])
    final_result = [mid]
    analyse_table = pd.DataFrame(final_result, index=[index])
    analyse_table.to_csv(file, mode='a', header=False)

    # analyse_table = pd.DataFrame(final_result, columns=['index', '参数', 'accuracy', 'balanced accuracy', 'sensitivity',
    #                                                     'specificity', 'auc', 'f1', 'precision', 'recall',
    #                                                     'tp', 'tn', 'fp', 'fn'])
