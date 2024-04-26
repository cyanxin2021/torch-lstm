import torch
from torch import nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json

import cross

seed = 42

INPUT_SIZE = 70
HIDDEN_LAYER_SIZE = 16
OUTPUT_SIZE = 1
NUM_LAYERS = 1
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 15
FOLD = 10

# 定义参数网格
param_grid = {
    # 'hidden_size': [8, 16, 32, 64],
    # 'num_layers': [1, 2],
    # 'learning_rate': [0.001, 0.01],
    # 'batch_size': [8, 16, 32, 64],
    # 'epoch': [i for i in range(10, 100, 10)]
    'hidden_size': [128],  # 6,32,64,128
    'num_layers': [1],
    'learning_rate': [0.001],  # 0.05,0.01,1e-3,1e-4
    'batch_size': [32],  # 8,16,32,64
    'epoch': [500],  # i for i in range(50, 300, 50)
    'dropout_prob': [0.2]
}

file = f'result/LSTM-test/4-20-fold{FOLD}.csv'
# file = 'result/LSTM-test/4-7-fold10.csv'

best_val_accuracies = []
best_epochs = []


key = eval(input('0. 训练模型\t1. 保存的模型\n'))


class LSTM(nn.Module):
    def __init__(self, input_size=70, hidden_layer_size=100, output_size=1, num_layers=1, dropout_prob=0.2):
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
        self.dropout = nn.Dropout(dropout_prob)
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
                       torch.zeros(self.num_layers, 1,
                                   self.hidden_layer_size))  # shape:(n_layers, batch_size, hidden_size)

        '''
        在 LSTM 模型中进行前向传播计算，并获取输出 lstm_out 以及最终的隐藏状态和细胞状态 h_n，h_c。
        '''
        lstm_out, (h_n, h_c) = self.lstm(input_x, hidden_cell)
        linear_out = self.linear(lstm_out.view(len(input_x), -1))  # = self.linear(lstm_out[:,-1,:])
        drop_out = self.dropout(linear_out)
        predictions = self.sigmoid(drop_out)
        return predictions


def means(list):
    mean_res = []
    for lt in list:
        m = np.mean(lt)
        mean_res.append(round(m, 4))
    return mean_res


def std(list):
    std_res = []
    for lt in list:
        s = np.std(lt)
        std_res.append(round(s, 4))
    return std_res


def main(epoch, hidden_size, num_layers, batch_size, lr, dropout_prob):
    # 设定随机种子以保证结果的可复现性
    torch.manual_seed(seed)
    np.random.seed(seed)

    train, label = cross.bold_train()

    mymodel = 'LSTM'  # 'SVM' vs 'DT'
    index = 'BOLD-' + mymodel
    sample = r'yes'

    best_accuracy = 0.0
    best_model = None

    balanced_res = []
    accuracy_res = []
    sensitivity_res = []
    specificity_res = []
    auc_res = []
    f1_res = []

    all_tp = []
    all_tn = []
    all_fp = []
    all_fn = []

    skf = StratifiedKFold(n_splits=FOLD, shuffle=True, random_state=seed)
    fold = 0
    for train_index, test_index in skf.split(train, label):
        print(f'#####fold {fold}')
        X_train, X_test = train[train_index], train[test_index]
        Y_train, Y_test = np.array(label)[train_index], np.array(label)[test_index]
        _, predicted_labels, true_labels, model = train_eval(X_train, Y_train, X_test, Y_test, epoch, fold, hidden_size,
                                                             num_layers, batch_size, lr, dropout_prob)

        accuracy, balanced, sensitivity, specificity, auc, f1, tp, tn, fp, fn = confusion_compute(predicted_labels,
                                                                                                  true_labels)
        accuracy_res.append(accuracy)
        balanced_res.append(balanced)
        sensitivity_res.append(sensitivity)
        specificity_res.append(specificity)
        auc_res.append(auc)
        f1_res.append(f1)

        all_tp.append(tp)
        all_tn.append(tn)
        all_fp.append(fp)
        all_fn.append(fn)

        print(f'Accuracy: {accuracy_res[fold]:.4f}')
        print(f'Balanced Accuracy: {balanced_res[fold]:.4f}')
        print(f'Sensitivity (TPR): {sensitivity_res[fold]:.4f}')
        print(f'Specificity (TNR): {specificity_res[fold]:.4f}')
        print(f'AUC: {auc_res[fold]:.4f}')
        print(f'F1 Score: {f1_res[fold]:.4f}')

        fold += 1

        # 保存最佳模型和参数
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # PATH = f'result/LSTM-test/state_dict_model_{epoch}.pth'
    # if not os.path.exists(PATH):
    #     torch.save(best_model.state_dict(), PATH)
    # print(f"Epoch: {epoch}, Best accuracy: {best_accuracy:.2f}")

    all_list = [accuracy_res, balanced_res, sensitivity_res, specificity_res,
                auc_res, f1_res, all_tp, all_tn, all_fp, all_fn]

    mean_result = means(all_list)

    std_result = std(all_list[:6])

    mid = tuple(str(mean_result[i]) + "±" + str(std_result[i]) for i in range(6)) + tuple(mean_result[6:]) + tuple(
        [sample, epoch, FOLD, hidden_size, num_layers, batch_size, lr, dropout_prob])

    final_result = [mid]
    if os.path.exists(file):
        analyse_table = pd.DataFrame(final_result, index=[index])
        analyse_table.to_csv(file, mode='a', header=False)
    else:
        analyse_table = pd.DataFrame(final_result, index=[index],
                                     columns=['accuracy', 'balanced accuracy', 'sensitivity',
                                              'specificity', 'auc', 'f1',
                                              'tp', 'tn', 'fp', 'fn', 'sample y/n', 'EPOCHS', 'FOLD', 'hidden_size',
                                              'num_layers', 'batch_size', 'lr', 'dropout_prob'])
        analyse_table.to_csv(file)
    return min(mean_result[:6]), best_model


# 获取验证得到的y_pred和y_true
def train_eval(X_train, Y_train, X_test, Y_test, num_epoch, f, hidden_size, num_layers, batch_size, lr, dropout_prob):
    rus = RandomUnderSampler(random_state=seed)
    # 欠采样
    X_train, Y_train = rus.fit_resample(X_train, Y_train)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 调整输入数据的格式
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(Y_train).float()  # 由于是分类任务，标签需要是 long 类型
    y_test = torch.from_numpy(Y_test).float()

    print(X_train.shape, '\n', y_train.shape)
    print(X_test.shape, '\n', y_test.shape)

    train_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_train, y_train),  # 封装进Data.TensorDataset()类的数据，可以为任意维度
        batch_size=batch_size,  # 每块的大小
        shuffle=True,  # 是否打乱顺序（打乱比较好）
    )
    test_loader = Data.DataLoader(
        dataset=Data.TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=True,  # 是否打乱顺序（打乱比较好）
    )

    # 建模三件套：loss, 优化，epoch
    model = LSTM(input_size=X_train.shape[2], hidden_layer_size=hidden_size, output_size=OUTPUT_SIZE,
                 num_layers=num_layers, dropout_prob=dropout_prob)  # 模型
    loss_func = nn.BCELoss()  # loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    print(model)  # 查看网络结构
    '''LSTM(
         (lstm): LSTM(1, 16)
         (linear): Linear(in_features=16, out_features=1, bias=True)
         (sigmoid): Sigmoid()
       )'''

    losses = []

    if key:
        path = r'result/fold-10/0.4599best_model.pth'
        model.load_state_dict(torch.load(path))
    else:
        # 训练模型
        for epoch in range(num_epoch):
            # 开始训练模型
            model.train()
            train_epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                y_pred = model(x)
                single_loss = loss_func(y_pred, y)

                single_loss.backward()
                optimizer.step()

                train_epoch_loss += single_loss.item()
            losses.append(train_epoch_loss)

            if epoch % 5 == 0:
                model.eval()
                val_epoch_loss = 0.0
                with torch.no_grad():
                    test_preds = []
                    for x, y in test_loader:
                        outputs = model(x)
                        single_loss = loss_func(outputs, y)
                        val_epoch_loss += single_loss.item()
                        preds = torch.round(torch.sigmoid(outputs))
                        test_preds.extend(preds.view(-1).tolist())
                    val_accuracy = accuracy_score(y_test.tolist(), test_preds)
                    print(
                        f"Fold {f}, Epoch {epoch}, Train Loss {train_epoch_loss / len(train_loader)}, Validation Loss {val_epoch_loss / len(test_loader)}, Validation Accuracy: {val_accuracy}")

                    # 记录最佳验证准确率和对应的epoch
                    if (not best_val_accuracies) or (val_accuracy > max(best_val_accuracies)):
                        best_val_accuracies.append(val_accuracy)
                        best_epochs.append(epoch)

    # 训练结束后外部测试
    model.eval()
    predicted_probs = []
    true_labels = []
    with torch.no_grad():
        # 这一部分的计算不需要跟踪计算
        for x, y in test_loader:
            outputs = model(x)
            predicted_probs.append(outputs.numpy())
            true_labels.append(y.numpy())

    predicted_probs = np.concatenate(predicted_probs).flatten()
    true_labels = np.concatenate(true_labels).flatten()

    # Convert probabilities to binary predictions using a threshold (e.g., 0.5)
    threshold = 0.5
    predicted_labels = (predicted_probs > threshold).astype(int)

    return losses, predicted_labels, true_labels, model


def draw(loss):
    plt.figure()
    plt.plot(range(1, len(loss) + 1), loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


def confusion_compute(y_pred, labels):
    accuracy_res = accuracy_score(labels, y_pred)
    balanced_res = balanced_accuracy_score(labels, y_pred)
    tn, fp, fn, tp = confusion_matrix(labels, y_pred).ravel()  # 让多维数组变成一维数组
    # print('tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp)
    sensitivity_res = tp / (tp + fn)  # 真阳性率
    specificity_res = tn / (tn + fp)  # 真阴性率
    auc = roc_auc_score(labels, y_pred)
    f1 = f1_score(labels, y_pred,
                  zero_division=1)  # f1_score函数的zero_division参数用于控制当分母为0时的行为。当该参数为0时，分母为0时返回0；当该参数为1时，分母为0时返回1；当该参数为其他数值时，分母为0时返回该数值。该参数的默认值为’warn’，表示当分母为0时会发出警告并返回0。当zero_division参数为1时，无论分母是否为0，F1分数都为1，即预测结果完全正确

    return accuracy_res, balanced_res, sensitivity_res, specificity_res, auc, f1, tp, tn, fp, fn


if __name__ == '__main__':
    # for i in range(1, 8):
    #     epoch = 50 * i
    #     samples = ['否,MDD:HC=169:89', '是,MDD:HC=89:89']
    #     for sample in samples:
    #         train(epoch, sample)

    # for i in range(1,8):
    #     epoch = 50 * i
    #     main(epoch)

    best_res = 0.0
    best_model = None
    best_params = None

    # 遍历参数网格
    for hidden_size in param_grid['hidden_size']:
        for num_layers in param_grid['num_layers']:
            for lr in param_grid['learning_rate']:
                for batch_size in param_grid['batch_size']:
                    for epoch in param_grid['epoch']:
                        for dropout_prob in param_grid['dropout_prob']:
                            print('-' * 50, f'\nepoch: {epoch}, hidden_size: {hidden_size}, num_layers: {num_layers}, '
                                            f'batch_size: {batch_size}, learning_rate: {lr}, dropout_prob: {dropout_prob}')
                            res, model = main(epoch, hidden_size, num_layers, batch_size, lr, dropout_prob)
                            print('min res: ', res)
                            if res > best_res:
                                best_res = res
                                best_model = model
                                best_params = {
                                    'hidden_size': hidden_size,
                                    'num_layers': num_layers,
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'epoch': epoch,
                                    'fold': FOLD,
                                    'dropout_prob': dropout_prob,
                                }

    dir = f'result/fold-{FOLD}/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # 保存最佳模型和参数
    best_model_path = os.path.join(dir, f'{best_res}best_model.pth')
    best_params_path = os.path.join(dir, f'{best_res}best_params.json')

    torch.save(best_model.state_dict(), best_model_path)
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f)

    print(f"Best value: {best_res}")
    print(f"Best parameters: {best_params}")
