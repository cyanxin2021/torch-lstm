import torch
from torch import nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score,f1_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import os

import cross

INPUT_SIZE = 70
HIDDEN_LAYER_SIZE = 16
OUTPUT_SIZE = 1
NUM_LAYERS = 1
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 15
FOLD = 10

class LSTM(nn.Module):
    def __init__(self, input_size=70, hidden_layer_size=100, output_size=1, num_layers=1):
        '''
        LSTM二分类任务
        :param input_size: 输入数据的维度
        :param hidden_layer_size: 隐层的数目
        :param output_size: 输出的个数
        '''
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_layer_size, num_layers=num_layers)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_x):
        '''
        input_x.view(len(input_x), 1, -1) 将输入数据 input_x 变形为一个新的形状，其中：
            第一维：保持原始大小，即 200（表示数据中的样本数）。
            第二维：将维度扩展为 1，因为 LSTM 模型的输入通常需要包含一个批次（batch）的信息。
            第三维：自动调整以适应新的形状，以满足输入到 LSTM 模型的要求。
        这个改变后的形状适合输入到 LSTM 模型的输入层。在 LSTM 中，输入数据的形状通常是 (batch_size, seq_len, input_size)，其中:
            batch_size: 表示每个批次中包含的数据样本数量。
            seq_len: 表示数据序列的长度。
            input_size: 表示每个时间步所包含的特征数。
        ⭐⭐⭐我的理解：将输入数据转换成 一个batch的样本数，每个样本序列长度，剩下的为特征数，也就是（样本数, 1, -1）
        '''
        input_x = input_x.view(len(input_x), 1,
                               -1)  # (batch, input_size)，其中input_x.shape=(1,5)，可以参照luojianmiaoshixiong的(input.size(0),input.size(1),-1)

        '''
        这行代码的作用是初始化 LSTM 的隐藏状态和细胞状态，使其变为 (1, 1, self.hidden_layer_size) 的形状，以便与输入数据相匹配。
        在这里，我们用 torch.zeros 来创建全零的张量作为初始隐藏状态和细胞状态。其中，第一个 1 表示层数，第二个 1 表示批次大小，
        而 self.hidden_layer_size 则表示每个隐藏状态的大小。
        n_layers=1表示我们使用的是单层 LSTM
        '''
        '''
        这个语句的作用是创建 LSTM 模型的初始隐藏状态和细胞状态。它包含两部分，分别对应隐藏状态（h）和细胞状态（c）。
        '''
        hidden_cell = (torch.zeros(self.num_layers, 1, self.hidden_layer_size),
                       torch.zeros(self.num_layers, 1, self.hidden_layer_size))  # shape:(n_layers, batch_size, hidden_size)

        '''
        在 LSTM 模型中进行前向传播计算，并获取输出 lstm_out 以及最终的隐藏状态和细胞状态 h_n，h_c。
        '''
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # = self.linear(lstm_out[:,-1,:])
        predictions = self.sigmoid(linear_out)
        return predictions


def lstm_run(X_train, X_test, y_train, y_test, epochs):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 调整输入数据的格式
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))


    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()  # 由于是分类任务，标签需要是 long 类型
    y_test = torch.from_numpy(y_test).float()

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=BATCH_SIZE,  # 每块的大小
        shuffle=True,  # 是否打乱顺序（打乱比较好）
        # num_workers=2,  # 多进程（multiprocess）来读数据
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=True,  # 是否打乱顺序（打乱比较好）
        # num_workers=2,  # 多进程（multiprocess）来读数据
    )

    # 建模三件套：loss, 优化，epoch
    model = LSTM(INPUT_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_SIZE, NUM_LAYERS)  # 模型
    loss_func = nn.BCELoss()  # loss
    # loss_func = nn.CrossEntropyLoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器
    prev_loss = 1000

    # 开始训练
    model.train()
    for epoch in range(epochs):
        for seq, labels in train_loader:
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            y_pred = model(seq)
            if y_pred.shape[0] == 1:
                y_pred = y_pred[0]
            else:
                y_pred = y_pred.squeeze()  # 压缩维度：得到输出，并将维度为1的去除
            # print('y_pred - ', y_pred, ',\tlabels - ', labels)
            single_loss = loss_func(y_pred, labels)
            # 若想获得类别，二分类问题使用四舍五入的方法即可：print(torch.round(y_pred))
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            # if single_loss < prev_loss:
            #     torch.save(model.state_dict(), 'lstm_model.pt')  # save model parameters to files
            #     prev_loss = single_loss
            #
            # if single_loss.item() < 1e-4:
            #     print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, epochs, single_loss.item()))
            #     print("The loss value is reached")
            #     break
            # else:
            #     print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, epochs, single_loss.item()))
    # 开始验证
    model.eval()
    total = 0
    correct = 0
    predicted = 0
    labels = 0
    '''
    总结with工作原理：
    （１）紧跟with后面的语句被求值后，返回对象的“–enter–()”方法被调用，这个方法的返回值将被赋值给as后面的变量；
    （２）当with后面的代码块全部被执行完之后，将调用前面返回对象的“–exit–()”方法。
    '''
    # with torch.no_grad():
    #     for inputs, labels in test_loader:
    #         # outputs = model(inputs)
    #         outputs = model(inputs).squeeze()
    #         predicted = (outputs > 0.5).float()
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    # with torch.no_grad():
    #     outputs = model(X_test).sequeeze()
    #     predicted = (outputs>0.5).float()
    #     total += y_test.size(0)
    #     correct += (predicted == y_test).sum().item()
    # accuracy = correct / total
    # print(f'测试精度: {accuracy * 100:.2f}%')

    predicted_probs = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted_probs.append(outputs.numpy())
            true_labels.append(labels.numpy())

    predicted_probs = np.concatenate(predicted_probs).flatten()
    true_labels = np.concatenate(true_labels).flatten()

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    predicted_labels = (predicted_probs > threshold).astype(int)

    # return predicted, labels
    # return predicted, y_test
    return predicted_labels,true_labels


def train(epoch, sample):
    train, label = cross.bold_train()
    # para = 'Optimize'  # 'none' vs 'Optimize'
    mymodel = 'LSTM'  # 'SVM' vs 'DT'
    index = 'BOLD-' + mymodel
    # sample = '否,MDD:HC=169:89'
    # sample = '是,MDD:HC=89:89'
    # file = './result/' + mymodel + '.csv'
    file = 'result/LSTM-test/3-13.csv'  # Epoch=50, 五折交叉验证

    rus = RandomUnderSampler(random_state=100)
    key = (sample == '是,MDD:HC=89:89')

    # ch = input("1. 状态序列\t2. 静态FC\t3. WPE矩阵\t4. 静态FC+WPE矩阵\t5. 状态序列+静态FC\t6. 状态序列+WPE矩阵\t7. 状态序列+WPE矩阵+静态FC\n")
    # if ch == '1':
    #     # BOLD
    #     train, label = bold_train()
    #     index = 'BOLD-' + mymodel
    # elif ch == '2':
    #     # FC的文件
    #     train, label = fc_train()
    #     index = 'FC-' + mymodel
    # elif ch == '3':
    #     # WPE的文件
    #     train, label = wpe_train()
    #     index = 'WPE-' + mymodel
    # elif ch == '4':
    #     # FC + WPE
    #     train, label = fc_wpe()
    #     index = 'FC+WPE-' + mymodel
    # elif ch == '5':
    #     # BOLD + FC
    #     train, label = bold_fc()
    #     index = 'BOLD+FC-' + mymodel
    # elif ch == '6':
    #     # BOLD + WPE
    #     train, label = bold_wpe()
    #     index = 'BOLD+WPE-' + mymodel
    # elif ch == '7':
    #     # BOLD + WPE
    #     train, label = bold_wpe_fc()
    #     index = 'BOLD+WPE+FC-' + mymodel

    # ch_ = input("\n是否需要欠采样: y or n\n")
    # if ch_ == 'y':
    #     rus = RandomUnderSampler(random_state=300)
    #     train, label = rus.fit_resample(train, label)
    #     sample = '欠采样,MDD:HC=89:89'

    balanced_res = []
    accuracy_res = []
    sensitivity_res = []
    specificity_res = []
    auc = []
    f1 = []

    all_tp = []
    all_tn = []
    all_fp = []
    all_fn = []



    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=500)
    fold = 0
    for train_index, test_index in skf.split(train, label):
        print('#####fold ', fold)
        X_train, X_test = train[train_index], train[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]

        # 欠采样
        if key:
            X_train, Y_train = rus.fit_resample(X_train, Y_train)



        y_pred, labels = lstm_run(X_train, X_test, Y_train, Y_test, epoch)

        accuracy_res.append(accuracy_score(labels,y_pred))
        balanced_res.append(balanced_accuracy_score(labels, y_pred))
        tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()  # 让多维数组变成一维数组
        # print('tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp)
        all_tp.append(tp)
        all_tn.append(tn)
        all_fp.append(fp)
        all_fn.append(fn)
        sensitivity_res.append(tp / (tp + fn))  # 真阳性率
        specificity_res.append(tn / (tn + fp))  # 真阴性率
        auc.append(roc_auc_score(labels, y_pred))
        f1.append(f1_score(labels, y_pred, zero_division=1))  # f1_score函数的zero_division参数用于控制当分母为0时的行为。当该参数为0时，分母为0时返回0；当该参数为1时，分母为0时返回1；当该参数为其他数值时，分母为0时返回该数值。该参数的默认值为’warn’，表示当分母为0时会发出警告并返回0。当zero_division参数为1时，无论分母是否为0，F1分数都为1，即预测结果完全正确
        # precision_res.append(precision_score(labels, y_pred))
        # recall_res.append(recall_score(labels, y_pred))
        # epoch = [(para, accuracy_res[fold], balanced_res[fold], sensitivity_res[fold], specificity_res[fold],
        #           auc[fold], f1[fold], precision_res[fold], recall_res[fold], tp, tn, fp, fn, sample)]
        # analyse_table = pd.DataFrame(epoch, index=[fold])
        # analyse_table.to_csv(file, mode='a', header=False)


        print(f'Accuracy: {accuracy_res[fold]:.4f}')
        print(f'Sensitivity (TPR): {sensitivity_res[fold]:.4f}')
        print(f'Specificity (TNR): {specificity_res[fold]:.4f}')
        print(f'AUC: {auc[fold]:.4f}')
        print(f'F1 Score: {f1[fold]:.4f}')
        fold += 1

    # final_result = [np.mean(accuracy_res), np.mean(balanced_res), np.mean(sensitivity_res), np.mean(specificity_res),
    #                 np.mean(auc), np.mean(f1), np.mean(precision_res), np.mean(recall_res), np.mean(all_tp),
    #                 np.mean(all_tn), np.mean(all_fp), np.mean(all_fn)]
    # final_result = [np.mean(accuracy_res), np.mean(balanced_res), np.mean(sensitivity_res), np.mean(specificity_res),
    #                 np.mean(auc), np.mean(f1), np.mean(all_tp), np.mean(all_tn), np.mean(all_fp), np.mean(all_fn)]
    # mid = tuple(round(i, 4) for i in final_result[:6]) + tuple(final_result[6:]) + tuple([sample, epoch])
    mean_result = [np.mean(accuracy_res), np.mean(balanced_res), np.mean(sensitivity_res), np.mean(specificity_res),
                    np.mean(auc), np.mean(f1), np.mean(all_tp), np.mean(all_tn), np.mean(all_fp), np.mean(all_fn)]

    mean_result = [round(i,4) for i in mean_result]

    std_result = [np.std(accuracy_res), np.std(balanced_res), np.std(sensitivity_res), np.std(specificity_res),
                  np.std(auc), np.std(f1)]
    std_result = [round(i,4) for i in std_result]

    mid = tuple(str(mean_result[i])+"±"+str(std_result[i]) for i in range(6)) + tuple(mean_result[6:]) + tuple([sample, epoch, FOLD])

    final_result = [mid]
    if os.path.exists(file):
        analyse_table = pd.DataFrame(final_result, index=[index])
        analyse_table.to_csv(file, mode='a', header=False)
    else:
        analyse_table = pd.DataFrame(final_result, index=[index], columns=['accuracy', 'balanced accuracy', 'sensitivity',
                                                            'specificity', 'auc', 'f1',
                                                            'tp', 'tn', 'fp', 'fn', 'sample y/n', 'EPOCHS', 'FOLD'])
        analyse_table.to_csv(file)

if __name__ == '__main__':
    # for i in range(1, 8):
    #     epoch = 50 * i
    #     samples = ['否,MDD:HC=169:89', '是,MDD:HC=89:89']
    #     for sample in samples:
    #         train(epoch, sample)
    sample = '是,MDD:HC=89:89'
    for i in range(1,8):
        epoch = 50 * i
        train(epoch,sample)
    # train(250,sample)

