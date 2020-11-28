'''
支持向量机实现 男女生分类
'''
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC 
from pr_config import My_count, My_cv_iterator, My_CV
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ori_data = pd.read_excel('作业数据_2020合成.xls')
np.random.seed(0)
# %%
used_feature = ['性别 男1女0','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量']
train_data = ori_data[used_feature]
train_data.dropna(inplace = True)
train_data = np.array(train_data)
train_data_feature = train_data[:,1:]
train_data_label = train_data[:,0]
mu = train_data_feature.mean(axis = 0)
sig = train_data_feature.std(axis = 0)
train_data_feature = (train_data_feature - mu)/sig # 标准化数据

# %% svm封装
def svmpredict(train_data,train_label,test_data,test_label, c,k = 'rbf', g = 'auto'):
    '''
    input
    c ： 惩罚系数
    k :  核函数 默认高斯核
    output
    predict : 预测结果
    ''' 
    svc = SVC(kernel=k,C=c,gamma = g, probability= True)
    svc = svc.fit(train_data,train_label)
    predict_label = svc.predict(test_data)
    predict_prob = svc.decision_function(test_data)
    fpr,tpr,_ = roc_curve(test_label,predict_prob)
    return predict_label,fpr,tpr
# %% 交叉验证
train_data,train_label,valid_data,valid_label = My_CV(train_data_feature, train_data_label, 5)
predict, fpr, tpr = svmpredict(train_data,train_label,valid_data,valid_label, c = 1, g = 2)
AUC = auc(fpr, tpr)
plt.plot(fpr,tpr,marker = 'o',label='ROC curve (area = {:.2f})' .format(AUC))
plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
recall=recall_score(valid_label,predict, average=None)
print("召回率",recall)
precision = precision_score(valid_label, predict, average= None)
print('准确率',precision)
My_count(valid_label,predict)
# %% 5折交叉验证
fold_num = 5
My_5_fold = My_cv_iterator(train_data_feature,train_data_label, fold_num)
C = [5, 2, 1, 0.1, 0.01]
se_mean = 0
sp_mean = 0
acc_mean = 0
C = 50
for step, train_data, valid_data in My_5_fold: # ROC曲线构造
    m,n = train_data.shape
    train_x = train_data[:,:n-1]
    train_y = train_data[:,-1]
    valid_x = valid_data[:,:n-1]
    valid_y = valid_data[:,-1]
    predict, fpr, tpr = svmpredict(train_x, train_y, valid_x, valid_y, c = C, k = 'rbf')
    AUC = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(step, AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with C = {}'.format(C))
    plt.legend(loc="lower right")
    print('\n****第{}折交叉验证, 惩罚系数为{}****'.format(step,C))
    print('**利用sklearn库进行结果统计**')
    recall=recall_score(valid_y,predict, average=None)
    print("召回率",recall)
    precision = precision_score(valid_y, predict, average= None)
    print('准确率',precision)
    print('**利用自写库进行结果统计**')
    se, sp, acc = My_count(valid_y,predict)
    se_mean += se
    sp_mean += sp
    acc_mean += acc
sp_mean = sp_mean/fold_num
se_mean = se_mean/fold_num
acc_mean = acc_mean/fold_num
print('\n平均SP为{}'.format(sp_mean))
print('\n平均SE为{}'.format(se_mean))
print('\n平均ACC为{}'.format(acc_mean))
# %%
