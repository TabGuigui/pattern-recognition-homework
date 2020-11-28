'''
决策树实现
'''
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_curve, auc
from sklearn import tree
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from pr_config import My_count, My_cv_iterator, My_CV
import graphviz
# %%
ori_data = pd.read_excel('作业数据_2020合成.xls')
choose = '籍贯'
if choose == '男女':
    print('true')
    used_feature = ['性别 男1女0','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量']
    train_data = ori_data[used_feature]
    train_data.dropna(inplace = True)
    train_data = np.array(train_data)
    train_data_feature = train_data[:,1:]
    train_data_label = train_data[:,0]
    mu = train_data_feature.mean(axis = 0)
    sig = train_data_feature.std(axis = 0)
    train_data_feature = (train_data_feature - mu)/sig # 标准化数据
if choose == '籍贯':
    used_feature = ['籍贯','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量']
    train_data = ori_data[used_feature]
    train_data.dropna(inplace = True)
    train_data['籍贯'].unique()
    encode = pd.factorize(train_data['籍贯'])
    train_data['籍贯'] = encode[0]
    train_data = np.array(train_data)
    train_data_feature = train_data[:,1:]
    train_data_label = train_data[:,0]
    mu = train_data_feature.mean(axis = 0)
    sig = train_data_feature.std(axis = 0)
    train_data_feature = (train_data_feature - mu)/sig # 标准化数据
# %% 封装决策树训练模型
def dtpredict(train_data,train_label,test_data,test_label, cri, depth, min_samples_leaf):
    clf = tree.DecisionTreeClassifier(criterion=cri, max_depth=depth, min_samples_leaf= min_samples_leaf,random_state= 1)
    
    clf.fit(train_data,train_label)
    predict = clf.predict(test_data)
    y_hat = clf.predict_proba(test_data)[:,1]
    # print(y_hat)
    fpr,tpr,_ = roc_curve(test_label,y_hat)
    dot_data = tree.export_graphviz(clf, class_names = ['female','male'],out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render('structure')
    # graph.render("iris")
    return predict,fpr,tpr

def dtpredict_multiclass(train_data,train_label,test_data,test_label, cri, depth, min_samples_leaf):
    clf = tree.DecisionTreeClassifier(criterion=cri, max_depth=depth, min_samples_leaf= min_samples_leaf,random_state= 1)
    
    clf.fit(train_data,train_label)
    predict = clf.predict(test_data)
    # fpr,tpr,_ = roc_curve(test_label,predict)
    return predict
# %% 交叉验证
train_data,train_label,valid_data,valid_label = My_CV(train_data_feature, train_data_label, 3)
if choose == '男女':
    predict, fpr, tpr, graph = dtpredict(train_data,train_label,valid_data,valid_label, cri = 'entropy', depth = 3, min_samples_leaf = 3)
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
if choose == '籍贯':
    predict = dtpredict_multiclass(train_data,train_label,valid_data,valid_label, cri = 'gini', depth = 10, min_samples_leaf = 2)
    recall=recall_score(valid_label,predict, average=None)
    
    print("召回率",recall)
    precision = precision_score(valid_label, predict, average= None)
    
    print('准确率',precision)
    recall = np.array(recall)
    recall_mean = recall.mean()
    precision = np.array(precision)
    precision_mean = precision.mean()
    print('平均召回率',recall_mean)
    print('平均准确率',precision_mean)
# %%
fold_num = 5
se_mean = 0
sp_mean = 0
acc_mean = 0
My_5_fold = My_cv_iterator(train_data_feature,train_data_label, fold_num)
max_depth = 10
for step, train_data, valid_data in My_5_fold:
    m,n = train_data.shape
    train_x = train_data[:,:n-1]
    train_y = train_data[:,-1]
    valid_x = valid_data[:,:n-1]
    valid_y = valid_data[:,-1]
    predict, fpr, tpr = dtpredict(train_x,train_y,valid_x,valid_y, cri = 'entropy', depth = max_depth, min_samples_leaf = 3)
    se, sp, acc = My_count(valid_y,predict)
    AUC = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker = 'o',label='ROC curve (area = {:.2f})' .format(AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with max depth = {}'.format(max_depth))
    plt.legend(loc="lower right")
    se_mean += se
    sp_mean += sp
    acc_mean += acc
plt.savefig('tree5')
sp_mean = sp_mean/fold_num
se_mean = se_mean/fold_num

acc_mean = acc_mean/fold_num
print('\n平均SE为{}'.format(se_mean))
print('\n平均SP为{}'.format(sp_mean))
print('\n平均ACC为{}'.format(acc_mean))
# %%
