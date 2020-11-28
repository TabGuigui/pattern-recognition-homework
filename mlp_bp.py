'''
多层感知器实现
'''
#%% 导入数据处理包以及原始数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ori_data = pd.read_excel('作业数据_2020合成.xls')
from pr_config import My_count, My_cv_iterator, My_bp_network
np.random.seed(0)
# %% 简单观察数据
used_feature = ['性别 男1女0','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量']
train_data = ori_data[used_feature]
train_data.head()
train_data.describe()
train_data.info()
#%% 数据处理
train_data.dropna(inplace = True)
train_data = np.array(train_data)
train_data_feature = train_data[:,1:]
train_data_label = train_data[:,0]
mu = train_data_feature.mean(axis = 0)
sig = train_data_feature.std(axis = 0)
train_data_feature = (train_data_feature - mu)/sig # 标准化数据
#%% 封装train与test
def My_train(net, feature, label, epoch, lr, valid_feature, valid_label):
    m = len(feature) # 训练集数量
    m_valid = len(valid_feature) # 验证集数量
    loss_history = []
    valid_loss_history = []
    for ep in range(epoch):
        loss_epoch = 0
        valid_loss_epoch = 0
        for i in range(m):
            net_input = feature[i]
            y = label[i]
            y_hat = net.forward(net_input)
            loss = 1/2*(y-y_hat)**2
            loss_epoch += float(loss)
            net.backward(lr,y_hat,y)
        for i in range(m_valid): # 观察验证集loss
            net_input = valid_feature[i]
            y = valid_label[i]
            y_hat = net.forward(net_input)
            loss = 1/2*(y-y_hat)**2
            valid_loss_epoch += float(loss)
        loss_history.append(loss_epoch/m)
        valid_loss_history.append(valid_loss_epoch/m_valid)
    return loss_history,valid_loss_history
def My_test(net, test_feature, test_label, is_predict = True):
    y_hat_list = []
    predict = []
    for i in range(len(test_label)):
        y_hat = net.forward(test_feature[i])
        y_hat_list.append(y_hat)
        if is_predict == True:
            if y_hat > 0.5:
                predict_y = 1
            else:
                predict_y = 0
            predict.append(predict_y)
    y_hat_list = np.array(y_hat_list).reshape(-1,1)
    predict = np.array(predict)
    return y_hat_list, predict      
#%% 训练
epoch = 10
fold_num = 5
My_5_fold = My_cv_iterator(train_data_feature,train_data_label, fold_num)
lr = 1
# fig = plt.figure(figsize= (10,10))
# a = fig.add_subplot(2,1,1)
# b = fig.add_subplot(2,1,2)
mlp_list = []
se_mean = 0
sp_mean = 0
acc_mean = 0
print('\n***{}折交叉验证结果, 学习率为{}****'.format(fold_num,lr))
for step, train_data, valid_data in My_5_fold: # 结果统计
    m,n = train_data.shape
    train_x = train_data[:,:n-1]
    train_y = train_data[:,-1]
    valid_x = valid_data[:,:n-1]
    valid_y = valid_data[:,-1]
    mlp = My_bp_network(5,10,1)
    mlp_list.append(mlp)
    train_loss,valid_loss = My_train(mlp, train_x, train_y, epoch, lr, valid_x, valid_y)
    x = list(range(epoch))
    # x = [x*150 for x in range(len(train_loss))]
    # a.plot(x, train_loss)
    # a.plot(x, valid_loss)
    # a.set_xlabel('epoch')
    # a.set_ylabel('loss')
    # a.set_title('训练集与验证集在{}epoch训练过程中的loss'.format(epoch))
    y_hat, predict = My_test(mlp, valid_x,valid_y )
    print('\n**第{}折结果**'.format(step))
    se, sp, acc = My_count(valid_y,predict)
    se_mean += se
    sp_mean += sp
    acc_mean += acc
    print('*利用sklearn库进行结果统计*')
    recall=recall_score(valid_y,predict, average=None)
    print("召回率",recall)
    precision = precision_score(valid_y, predict, average= None)
    print('准确率',precision)
    fpr, tpr, thresholds = roc_curve(valid_y,y_hat)
    AUC = auc(fpr, tpr)
    # b.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(step,AUC))
    # b.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    # b.set_title('Receiver operating characteristic with lr = {}'.format(lr))
    # b.set_xlabel('False Positive Rate')
    # b.set_ylabel('True Positive Rate')
    # b.legend(loc="lower right")
    plt.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(step,AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    plt.title('Receiver operating characteristic with lr = {}'.format(lr))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    # a.legend(('10 trainloss','10 validloss', '1 trainloss', '1 validloss', '0.1 trainloss', '0.1 validloss', '0.01 trainloss','0.01 validloss', '0.001trainloss', '0.001validloss'))
sp_mean = sp_mean/fold_num
se_mean = se_mean/fold_num
acc_mean = acc_mean/fold_num
print('\n平均SE为{}'.format(se_mean))
print('\n平均SP为{}'.format(sp_mean))
print('\n平均ACC为{}'.format(acc_mean))
#%% 画ROC曲线
# My_5_fold = My_cv_iterator(train_data_feature,train_data_label, 5)
# lr = [10, 1, 0.1, 0.01, 0.001]
# for step, train_data, valid_data in My_5_fold: # ROC曲线构造
#     m,n = train_data.shape
#     train_x = train_data[:,:n-1]
#     train_y = train_data[:,-1]
#     valid_x = valid_data[:,:n-1]
#     valid_y = valid_data[:,-1]
#     mlp = My_bp_network(5,5,1)
#     train_loss,valid_loss = My_train(mlp, train_x, train_y, epoch, lr[step-1], valid_x, valid_y)
#     predict = My_test(mlp, valid_x,valid_y )
#     fpr, tpr, thresholds = roc_curve(valid_y,predict)
#     AUC = auc(fpr, tpr)
#     plt.plot(fpr,tpr,marker = 'o',label='ROC curve (area = {:.2f}) with lr = {}' .format(AUC,lr[step-1]))
#     plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")

# %%
epoch = 100
fold_num = 5
My_5_fold = My_cv_iterator(train_data_feature,train_data_label, fold_num)
fig = plt.figure(figsize= (10,10))
a = plt
# b = fig.add_subplot(2,1,2)
mlp_list = []
# print('\n***{}折交叉验证结果, 学习率为{}****'.format(fold_num,lr))
lr = [100, 1,0.1,0.0001]
for step, train_data, valid_data in My_5_fold: # 结果统计
    if step <= 4:
        m,n = train_data.shape
        train_x = train_data[:,:n-1]
        train_y = train_data[:,-1]
        valid_x = valid_data[:,:n-1]
        valid_y = valid_data[:,-1]
        mlp = My_bp_network(5,10,1)
        mlp_list.append(mlp)
        print(lr[step-1])
        train_loss,valid_loss = My_train(mlp, train_x, train_y, epoch, lr[step-1], valid_x, valid_y)
        x = list(range(epoch))
        # x = [x*150 for x in range(len(train_loss))]
        a.plot(x, train_loss)
        a.plot(x, valid_loss)
        a.xlabel('epoch')
        a.ylabel('loss')
        a.title('训练集与验证集在{}epoch训练过程中的loss'.format(epoch))
        y_hat, predict = My_test(mlp, valid_x,valid_y )
        # print('\n**第{}折结果**'.format(step))
        # My_count(valid_y,predict)
        # print('*利用sklearn库进行结果统计*')
        # recall=recall_score(valid_y,predict, average=None)
        # print("召回率",recall)
        # precision = precision_score(valid_y, predict, average= None)
        # print('准确率',precision)
        # fpr, tpr, thresholds = roc_curve(valid_y,y_hat)
        # AUC = auc(fpr, tpr)
        # b.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(step,AUC))
        # b.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
        # b.set_title('Receiver operating characteristic with lr = {}'.format(lr))
        # b.set_xlabel('False Positive Rate')
        # b.set_ylabel('True Positive Rate')
        # b.legend(loc="lower right")

a.legend(('100 trainloss','100 validloss', '1 trainloss', '1 validloss', '0.1 trainloss', '0.1 validloss','0.0001trainloss', '0.0001validloss'))
# a.legend(('1train','1valid','0.1train','0.1valid','0.0001train','0.0001valid'))
a.savefig('10')
# %%
