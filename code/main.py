#!/usr/bin/python3
# -*- coding:utf-8 -*-

# 基于Python实现KNN算法实现手写字识别
# 学生：张文浩
# 学号：20114210433
# Github: Zeaulo
# -在完成作业的同时也将作业相关的所有源码上传到了我的Github ： https://github/Zeaulo/Robust_EnsembleCKNNk
# -采用MIT开源许可协议

# 本次题目中所提出的最佳模型：Ensemble_CKNNk
# -说明：将预先训练好的CNN模型的中间层作为特征提取器，并且寻找到最佳k值后进行设置，再使用KNN算法进行拟合并推理，最后该模型与CNN模型集成，进行不等权的概率贡献（CNN：CKNNk = 4:6），得到最终的预测结果
# 具体过程如下：
# 1.特征提取过程：针对训练集的所有样本，使用3x3卷积核卷积一次提取特征，再将其拉平至一维，得到每个样本的表征向量embedding
# 2.拟合过程：    针对训练集的所有样本，将得到的表征向量作为特征，再输入于KNN算法进行拟合
# 3.推理过程：    针对测试集的所有样本，输入进拟合完成的KNN算法，得到预测结果

# 对比实验 - 与纯粹的KNN算法以及精调过的KNN算法进行对比：
# 1.KNN - 直接使用kNN算法
# 2.KNNk：KNN+寻找最佳k值 - 使用k折交叉验证的方法将训练集分割成训练集和验证集，并通过遍历的方法在验证集上测试在多少个点参与投票的情况下，KNN算法的准确率多少，从而找到最佳的k值。（确定最佳的K值，能够使得KNN算法的准确率提高）
# 3.CNN - 模型结构：卷积层->批归一层->激活层->LSTM层->全连接层
# 4.CKNN - 将预先训练好的CNN模型的中间层作为特征提取器，再使用KNN算法进行拟合并推理
# 5.Ensemble_CKNN - 模型集成，将CNN和CKNN的预测结果进行集成，得到最终的预测结果
# 6.Ensemble_CKNNk - 模型集成，将CNN和CKNNk的预测结果进行集成，得到最终的预测结果

# 不使用数据增强的实验结果：
# -KNN算法的准确率为：0.9851380042462845
# -KNNk算法的准确率为：0.9893842887473461，最佳的k值为：1
# -CNN算法的准确率为：0.9968152866242038
# -CKNN算法的准确率为：0.9904458598726115
# -CKNNk算法的准确率为：0.9936305732484076
# -Ensemble_CKNN算法的准确率为：0.9978768577494692
# -Ensemble_CKNNk算法的准确率为：1.0
# 模型：Ensemble_CKNNk > Ensemble_CKNN >   CNN   >   CKNNk   >   CKNN   >   KNNk   >   KNN
# 分数：     1.0             0.9978       0.9968    0.9936      0.9904     0.9893    0.9851

# 经过数据增强后的实验结果 ：
# -KNN算法的准确率为：0.9925690021231423
# -KNNk算法的准确率为：0.9936305732484076，最佳的k值为：1
# -CNN算法的准确率为：0.9968152866242038
# -CKNN算法的准确率为：0.9904458598726115
# -CKNNk算法的准确率为：0.9936305732484076
# -Ensemble_CKNN算法的准确率为：0.9978768577494692
# -Ensemble_CKNNk算法的准确率为：1.0
# 模型：Ensemble_CKNNk > Ensemble_CKNN >   CNN   >   CKNNk   =   KNNk   >  KNN   >   CKNN
# 分数：      1.0            0.9978       0.9968     0.9936     0.9936    0.9925    0.9904

# 结论：
# 1.数据增强的策略是有效的。若使用数据增强，能直接令KNN，KNNk模型的得分均得到提升，分别从0.9851、0.9893，提升至0.9925、0.9936。
# 2.CKNN模型与CNN模型能够很好地集成，并将二者原本的性能表现进一步地提高。同时，Ensemble_CKNNk模型在该数据集中的准确率达到了SOTA，为1.0。
# 3.将经过卷积提取的特征用于KNN模型的拟合中，能够进一步发挥KNN模型的性能，使拟合后的KNN模型有更强的泛化性。
# 4.能够使用k折交叉验证来确定在多少个点参与投票的情况下，KNN模型的表现能比sklearn内默认的参数更好。
# 5.卷积神经网络比单模型的KNN性能要更好。


# 导入依赖包
import os
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# 超参数设置
class opt:
    seed               = 100 # 设置随机种子，方便实验复现
    batch_size         = 64 # 每次训练输入64条样本至GPU/CPU
    set_epoch          = 30 # 训练30轮
    early_stop         = 15 # 早停机制，若15轮内损失不再降低则停止训练
    consideration_num  = 5 # 每训练五个模型，则对所训练的五个模型进行挑选，将表现最稳定的模型进行保存
    learning_rate      = 1e-3 # 设置学习率为0.001
    weight_decay       = 2e-6 # L2正则化
    device             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_num            = 1 # 使用1张显卡进行训练
    use_data_au        = True # 是否使用数据增强
    check_error        = False # 检验模型对哪些测试样本预测错误

# 模型定义
# 首先，使用3x3卷积核共84个（第一次用68个，第二次用16个）对原始图像卷积两次，再输进入批归一化层、激活层并拉平得到一维的卷积特征。
# 然后，将一维卷积特征输入至一层的LSTM当中（此举主要是挑选一维卷积特征中重要的部分），这里设置LSTM隐藏层的维度大小为256。
# 最后，再将LSTM最后一个序列状态输入至全连接层中进行训练。
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 已知输入为32x32的灰度图像，输出为10个类别标签的概率
        pic_size    = 32
        # 使用68个3x3的卷积核
        kernel1     = 3 # 3
        kernel1_num = 68 # 68个输出通道
        # 只经过卷积层、批归一层和激活层，为了保证精度而不经过池化层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=kernel1_num, kernel_size=kernel1, stride=1, padding=0),
            nn.BatchNorm2d(kernel1_num),
            nn.ReLU()
        )

        # 使用12个3x3的卷积核
        kernel2     = 3 # 3
        kernel2_num = 16 # 16个输出通道
        # 只经过卷积层、批归一层和激活层，为了保证精度而不经过池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=kernel1_num, out_channels=kernel2_num, kernel_size=kernel2, stride=1, padding=0),
            nn.BatchNorm2d(kernel2_num), 
            nn.ReLU()
        )
        # 使用一个3x3的卷积核对原始图片进行二维卷积，输出1通道的二维特征再拉平为1维，即可得到图片的表征
        self.conve = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)

        hidden_size = 256 #256
        # 计算输出进LSTM的维度大小
        field1_len  = int(pic_size+1-kernel1)
        flat_len  = kernel2_num*int(field1_len+1-kernel2)**2
        # LSTM - 用于选取经卷积后最有帮助的特征
        self.lstm   = nn.LSTM(input_size=flat_len, hidden_size=hidden_size)
        self.act    = nn.Tanh() # 激活函数
        self.classifier = nn.Linear(hidden_size, 10) # 分类器

    def forward(self, x):
        # 增加一个维度，符合进行卷积的输入格式
        x = x.unsqueeze(1)

        # embedding - 使用1个3x3的卷积核对原始图片进行一次卷积
        embedding = self.conve(x)
        # embedding - 并且将所有维度拉平为一维
        embedding = embedding.reshape([embedding.shape[0], embedding.shape[1]*embedding.shape[2]*embedding.shape[3]])

        # (原始)bx1x32x32 -> bx68x30x30 [批大小, 通道数, 图像长度, 图像宽度]
        x = self.conv1(x)
        # bx68x30x30 -> bx12x28x28
        x = self.conv2(x)
        # bx12x28x28 -> bx9408 拉平为1维
        x = x.reshape([x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]])   # reshape为一维向量  
        # bx9408 -> bx256 使用LSTM学习重要的特征，并输出256维的特征
        x,_= self.lstm(x)
        # 256维的特征经过激活后再输入至分类器中
        x = self.act(x) 
        # bx256 -> bx10 使用分类器(全连接层)针对256维的特征进行学习
        x = self.classifier(x)
        return x, embedding

# KNN：- 直接使用KNN算法
def KNN(x_train, y_train, x_test, y_test):
    model   = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred  = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    y_pred_prob  = model.predict_proba(x_test)
    return score, y_pred_prob

# KNNk：KNN+寻找最佳k值
def KNNk(x_train, y_train, x_test, y_test):
    # K折交叉验证
    fold5_x = KFold(n_splits=12, shuffle=True, random_state=100)
    fold_best_scores = [] 
    fold_best_k = []
    for train_index, val_index in fold5_x.split(x_train, y_train):
        x_train1, y_train1 = x_train[train_index], y_train[train_index]
        x_val, y_val = x_train[val_index], y_train[val_index]
        # 给k值设置遍历范围，这里从1~10去遍历
        k_list = [i for i in range(1, 11)]
        scores_list = []
        # 寻找最佳的k值
        for k in k_list:
            model   = KNeighborsClassifier(n_neighbors=k)
            # 使用train1来拟合KNN模型
            model.fit(x_train1, y_train1)
            # 再使用val来验证
            y_pred  = model.predict(x_val)
            # 评估准确度 - acc = 模型推理正确的val样本数 / val总样本数
            score = accuracy_score(y_val, y_pred)
            # 记录分数
            scores_list.append(score)
        # 找到每一个fold的最佳k值 - 挑出分数最高的模型对应的k值
        fold_best_scores.append(max(scores_list))
        fold_best_k.append(k_list[scores_list.index(max(scores_list))])

    # 选择分数最高
    best_k = fold_best_k[fold_best_scores.index(max(fold_best_scores))]
    # 使用最佳的k值，再使用为分割前的训练集train来对KNN模型重新进行拟合
    model   = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(x_train, y_train)
    # 使用test数据集来进行测试
    y_pred  = model.predict(x_test)
    # 得到KNN模型的最终得分
    score = accuracy_score(y_test, y_pred)
    y_pred_prob  = model.predict_proba(x_test)
    return score, y_pred_prob, best_k

# 检错 - 检验测试集里面是否有打错标签的样本或混淆不清的样本
def check_error_samples(samples, pred, true):
    fixed_index = -1
    fixed_value = 0
    for i in range(pred.shape[0]):
        # 准确定位到文件路径，获取文件名的手段
        if fixed_index  == true[i]:
            fixed_value += 1
        else:
            fixed_index  = true[i]
            fixed_value  = 0
        # 开始对比
        if pred[i] != true[i]:
            img = samples[i]
            print(f'\n文件名：{fixed_index}_{fixed_value}.txt\n预测标签：{pred[i]}\n真实标签：{true[i]}')
            plt.subplot(111)
            plt.imshow(img, cmap='gray')
            plt.title(f'file_name:{fixed_index}_{fixed_value}.txt, Prediction:{pred[i]}, True class:{true[i]}')
            plt.show()


# 随机种子设置
seed = opt.seed
torch.seed = seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# 一、数据预处理
def DataPreprocess():
    dataset_path         = '../dataset'                    # 定位数据集的相对路径
    train_path           = dataset_path+'/trainingDigits'  # 定位训练集路径
    test_path            = dataset_path+'/testDigits'      # 定位测试集路径
    # 将数据集转化为numpy数组，feature为样本特征，label为样本标签
    def dataset_process(path, name):
        sample_file      = os.listdir(path)
        # 重新排列 - 方便后续检错处理中的文件定位操作
        numbers = {f'{i}':[] for i in range(10)}
        for char in sample_file:
            numbers[char.split('_')[0]] += [char]
        sample_file = []
        for i in range(10):
            i = str(i)
            numbers[i].sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
            sample_file += numbers[i]
        # 读取所有样本
        data             = []
        labels           = []
        for file in sample_file:
            # 读取单个样本
            pre_process  = open(f'{path}/{file}', 'r').read().replace('0', '0 ').replace('1', '1 ').strip()
            temp         = open('temp.txt', 'w')
            temp.write(pre_process)
            temp.close()
            sample       = np.loadtxt('temp.txt')
            # 将单个样本的数据和标签分别存储
            data.append(sample)
            labels.append(int(file.split('_')[0]))
        # 将所有样本的数据和标签转化为numpy数组
        data      = np.array(data)
        labels    = np.array(labels)
        # 保存数据
        np.save(f'{name}_data.npy'  , data)
        np.save(f'{name}_labels.npy', labels)
    # 判断是否是第一次处理，若是则进行数据预处理，若不是则直接读取所保存的数据
    if not os.path.exists(f'train_data.npy'):
        dataset_process(train_path, 'train')
        dataset_process(test_path,  'test')
    train_data          = torch.from_numpy(np.load('train_data.npy'  , allow_pickle=True)).float()
    train_label         = torch.from_numpy(np.load('train_labels.npy', allow_pickle=True)).long()
    test_data           = torch.from_numpy(np.load('test_data.npy'   , allow_pickle=True)).float()
    test_label          = torch.from_numpy(np.load('test_labels.npy' , allow_pickle=True)).long()
    train_dataset       = torch.utils.data.TensorDataset(train_data  , train_label)
    test_dataset        = torch.utils.data.TensorDataset(test_data   , test_label)
    return train_dataset, test_dataset

# 数据增强 - 1.膨胀 2.腐蚀 3.旋转
# 使用数据增强后，训练集从1934条样本扩增至7736条样本，共增加了5804条样本
def data_au(x_train, y_train):
    # 进行膨胀和腐蚀操作
    for i in range(len(x_train)):
        img = x_train[i]
        # 1.膨胀
        img1 = cv2.dilate(img, np.ones([1, 1], np.uint8), iterations=1)
        # 2.腐蚀
        img2 = cv2.erode (img, np.ones([2, 2], np.uint8), iterations=1)
        # 3.旋转 - 随机逆时针旋转5~60°
        M1   = cv2.getRotationMatrix2D((16, 16), random.randint(5, 60), 1) 
        img3 = cv2.warpAffine(img, M1, (32, 32))
        img3 = np.round(img3)
        x_train = np.concatenate([x_train, [img1], [img2], [img3]], axis=0)
        y_train = np.concatenate([y_train, [y_train[i]]*3], axis=0)
    return x_train, y_train

train_dataset, test_dataset = DataPreprocess()
# 打印数据集信息
print('-数据集信息：')
print(f'-训练集样本数：{len(train_dataset)}，测试集样本数：{len(test_dataset)}')
train_labels = len(set(train_dataset.tensors[1].numpy()))
test_labels  = len(set(test_dataset.tensors[1].numpy()))
# 看一下各类样本数目是否均衡
print(f'-训练集的标签种类个数为：{train_labels}，测试集的标签种类个数为：{test_labels}')
numbers = [0] * train_labels
for i in train_dataset.tensors[1].numpy():
    numbers[i] += 1
print(f'-训练集各种类样本的个数：')
for i in range(train_labels):
    print(f'-数字{i}的样本个数为：{numbers[i]}')

# 从训练集中随机抽取一个样本来展示
# sample_data, sample_label = train_dataset[random.randint(0, len(train_dataset)-1)]
# print(f'-现在所展示的数字为：{sample_label}，它以及这个数据集当中所有样本的尺寸皆为：{sample_data.shape[0]}x{sample_data.shape[1]}')
# cv2.imshow('sample', sample_data.numpy())
# cv2.waitKey(0)

# 测试直接使用KNN算法拟合，能有多少的准确率
x_train = train_dataset.tensors[0].numpy()
y_train = train_dataset.tensors[1].numpy()

# 数据增强
if opt.use_data_au:
    x_train, y_train = data_au(x_train, y_train)

x_test  = test_dataset.tensors[0].numpy()
y_test  = test_dataset.tensors[1].numpy()

# 超参数设置
batch_size         = opt.batch_size
set_epoch          = opt.set_epoch
early_stop         = opt.early_stop
consideration_num  = opt.consideration_num
learning_rate    = opt.learning_rate
weight_decay     = opt.weight_decay
device          = opt.device
gpu_num            = opt.gpu_num
check_error       = opt.check_error

# 迭代器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 二、模型预训练
def model_pretrain():
    # 判断最佳模型是否已经存在，若存在则直接读取，若不存在则进行预训练
    if os.path.exists('checkpoints/best_model.pth'):
        try:
            best_model = CNN()
            best_model.load_state_dict(torch.load('checkpoints/best_model.pth'))
            return best_model
        except:
            pass
    else:
        pass
    model_save_dir  = 'checkpoints'
    model_name      = 'CNN'
    result_save_dir = 'results'

    # 模型加载
    model = CNN().to(device)

    # Optimizer loading
    if device     != 'cpu' and gpu_num > 1:
        optimizer = torch.optim.NAdam(model.module.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=list(range(gpu_num)))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # 损失函数加载
    loss_func = nn.CrossEntropyLoss()

    # 模型训练
    # -模型训练这部分的代码来源于我的Github项目：Torcheasy - https://github.com/Zeaulo/Torcheasy
    best_epoch         = 0
    best_train_loss    = 100000
    train_acc_list     = []
    train_loss_list    = []
    best_model_loss    = 100000
    models_list        = [i for i in range(consideration_num)]
    start_time         = time.time()

    for epoch in range(set_epoch):
        model.train()
        train_loss = 0
        train_acc = 0
        for i, (x, y) in enumerate(train_loader):
            x                   = x.to(device)
            y                   = y.to(device)
            outputs, embedding  = model(x)
            
            loss                = loss_func(outputs, y)
            train_loss         += loss.item()
            optimizer.zero_grad()
            loss.backward()

            if device != 'cpu' and gpu_num > 1:
                optimizer.module.step()
            else:
                optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)

            train_acc   += (predicted == y).sum().item()
        
        average_mode = 'macro'
        train_f1     = metrics.f1_score(y.cpu(), predicted.cpu(), average=average_mode)
        train_pre    = metrics.precision_score(y.cpu(), predicted.cpu(), average=average_mode)
        train_recall = metrics.recall_score(y.cpu(), predicted.cpu(), average=average_mode)

        train_loss /= len(train_loader)
        train_acc  /= len(train_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        # check_error_samples(x.cpu(), predicted.cpu(), y.cpu())

        print('-'*50)
        print('Epoch [{}/{}]\n Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch + 1, set_epoch, train_loss, train_acc))
        print('Train-f1: {:.4f}, Train-precision: {:.4f} Train-recall: {:.4f}'.format(train_f1, train_pre, train_recall))

        # Choose the best model for saving1
        # (epoch+1)%save_num -> Replace the old model in the list
        models_list[(epoch+1)%consideration_num] = model
        if epoch+1 >= consideration_num:
            # -save_num:-1 -> The model score list is always in the last few numbers
            models_list_loss = train_loss_list[-consideration_num:-1]
            # model_loss -> model_loss is the average loss of the model list
            # The purpose is to find the most stable model list
            model_loss = np.mean(models_list_loss)
            if model_loss <= best_model_loss:
                best_model_loss = model_loss
                # Choose the best model in the model list -> The best model in the most stable model list
                perfect = np.argmin(models_list_loss)
                print(f'-> best_model_loss {model_loss} has accessed, saving model...')
                if device == 'cuda' and gpu_num > 1:
                    torch.save(models_list[perfect].module.state_dict(), f'{model_save_dir}/best_model.pth')
                else:
                    torch.save(models_list[perfect].state_dict(), f'{model_save_dir}/best_model.pth')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_epoch = epoch + 1
        
        # Early stopping
        if epoch+1 - best_epoch == early_stop:
            print(f'{early_stop} epochs later, the loss of the validation set no longer continues to decrease, so the training is stopped early.')
            end_time = time.time()
            print(f'Total time is {end_time - start_time}s.')
            break

        # Draw the accuracy and loss function curves of the training set and the validation set
        plt.figure()
        plt.plot(range(1, len(train_acc_list) + 1), train_acc_list, label='train_acc')
        plt.legend()
        plt.savefig(f'{result_save_dir}/{model_name}_acc.png')
        plt.close()
        plt.figure()
        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, label='train_loss')
        plt.legend()
        plt.savefig(f'{result_save_dir}/{model_name}_loss.png')
        plt.close()
    best_model = CNN()
    best_model.load_state_dict(torch.load('checkpoints/best_model.pth'))
    return best_model

best_model = model_pretrain()

# 三、模型推理
def model_predict(model):
    model.to('cpu')
    y_train_labels   = []
    train_embedding = []
    test_embedding  = []
    test_outputs    = None
    # 将训练集的数据输入进模型，得到经过卷积提取后的特征
    for i, (data, label) in enumerate(train_loader):
        data = data.to('cpu')
        _, embedding = model(data)
        if i == 0:
            train_embedding = embedding
            y_train_labels  = label
        else:
            train_embedding = torch.cat([train_embedding, embedding], dim=0)
            y_train_labels  = torch.cat([y_train_labels, label], dim=0)
    train_embedding = train_embedding.detach().numpy()

    # 将测试集的数据输入进模型，得到经过卷积提取后的特征
    for i, (data, label) in enumerate(test_loader):
        data = data.to('cpu')
        outputs, embedding = model(data)
        if i == 0:
            test_embedding = embedding
            test_outputs = outputs
        else:
            test_embedding = torch.cat([test_embedding, embedding], dim=0)
            test_outputs = torch.cat([test_outputs, outputs], dim=0)
    test_embedding = test_embedding.detach().numpy()

    # 计算CNN模型的准确率分数
    y_pred     = torch.argmax(test_outputs, dim=1).detach().numpy()
    CNN_prob   = test_outputs.detach().numpy()
    CNN_score  = accuracy_score(y_test, y_pred)
    # 计算CKNN模型和CKNNk的准确率分数
    CKNN_score,  CKNN_prob     = KNN(train_embedding, y_train_labels, test_embedding, y_test)
    CKNNk_score, CKNNk_prob, _ = KNNk(train_embedding, y_train_labels, test_embedding, y_test)
    # 计算Ensemble_CKNN模型的准确率分数
    Ensemble_CKNN_prob   = 0.4*CNN_prob + 0.6*CKNN_prob
    Ensemble_CKNN_pred   = np.argmax(Ensemble_CKNN_prob, axis=1)
    Ensemble_CKNN_score  = accuracy_score(y_test, Ensemble_CKNN_pred)
    # 计算Ensemble_CKNNk模型的准确率分数
    Ensemble_CKNNk_prob   = 0.4*CNN_prob + 0.6*CKNNk_prob
    Ensemble_CKNNk_pred   = np.argmax(Ensemble_CKNNk_prob, axis=1)
    Ensemble_CKNNk_score  = accuracy_score(y_test, Ensemble_CKNNk_pred)  

    # 检错
    if check_error:
        check_error_samples(x_test, Ensemble_CKNNk_pred, y_test)

    return CNN_score, CKNN_score, CKNNk_score, Ensemble_CKNN_score, Ensemble_CKNNk_score

# 对比实验
# 数据需要reshape为一维
x_train_knn = x_train.reshape([x_train.shape[0], x_train.shape[1]*x_train.shape[2]])
x_test_knn  = x_test.reshape([x_test.shape[0], x_test.shape[1]*x_test.shape[2]])

KNN_score, _  = KNN(x_train_knn, y_train, x_test_knn, y_test)
KNNk_score, _, k = KNNk(x_train_knn, y_train, x_test_knn, y_test)
CNN_score, CKNN_score, CKNNk_score, Ensemble_CKNN_score, Ensemble_CKNNk_score = model_predict(best_model)
print(f'-KNN算法的准确率为：{KNN_score}')
print(f'-KNNk算法的准确率为：{KNNk_score}，最佳的k值为：{k}')
print(f'-CNN算法的准确率为：{CNN_score}')
print(f'-CKNN算法的准确率为：{CKNN_score}')
print(f'-CKNNk算法的准确率为：{CKNNk_score}')
print(f'-Ensemble_CKNN算法的准确率为：{Ensemble_CKNN_score}')
print(f'-Ensemble_CKNNk算法的准确率为：{Ensemble_CKNNk_score}')
