'''
author : tabgui
time : 2020.11.12
content : pca实现男女数据集特征降维
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
from numpy.linalg import det
from sklearn import preprocessing
from sklearn.decomposition import PCA
from svm import svmpredict
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ori_data = pd.read_excel('作业数据_2020合成.xls')
from pr_config import My_count, My_cv_iterator, My_bp_network,My_CV
np.random.seed(0)
random.seed(1)
# %% 简单观察数据
used_feature = ['性别 男1女0','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量','喜欢运动']
train_data = ori_data[used_feature]
train_data.dropna(inplace = True)
train_data["身高(cm)"]=preprocessing.scale(train_data["身高(cm)"]) 
train_data["体重(kg)"]=preprocessing.scale(train_data["体重(kg)"]) 
train_data["50米成绩"]=preprocessing.scale(train_data["50米成绩"]) 
train_data["鞋码"]=preprocessing.scale(train_data["鞋码"]) 
train_data['肺活量'] = preprocessing.scale(train_data['肺活量'])
train_data['喜欢运动'] = preprocessing.scale(train_data['喜欢运动'])
train_data.head()
train_data.describe()
train_data.info()

#%%
num = 5
np.random.seed(1)
train_data = np.array(train_data)
train_data_feature = train_data[:,1:]
train_data_label = train_data[:,0]
# mu = train_data_feature.mean(axis = 0)
# sig = train_data_feature.std(axis = 0)
# train_data_feature = (train_data_feature - mu)/sig # 标准化数据
label = train_data_label.copy()
index_male = np.where(label == 1)
index_female = np.where(label == 0)
male_data = train_data_feature.copy()[index_male]
female_data = train_data_feature.copy()[index_female]
pca = PCA(n_components=6)
pca.fit(train_data_feature)

train_data_feature = pca.transform(train_data_feature)

#%%
# plt.plot(np.array(range(1,6)),pca.explained_variance_,label="方差值")
plt.scatter(np.array(range(1,7)),pca.explained_variance_)
plt.title('各主成分的方差值')
plt.xlabel("主成分")
plt.ylabel("方差值")
#     print(newData)
print('explained_variance_ratio: ', pca.explained_variance_ratio_)
print('explained_variance: ', pca.explained_variance_)
print('n_components: ', pca.n_components_)
plt.savefig("主成分-方差.png")
plt.show()
#%%
from sklearn.model_selection import KFold
kf = KFold(n_splits=fold_num)
fold_num = 5
se_mean = 0
sp_mean = 0
acc_mean = 0
C = 10
pca_list = [PCA(n_components='mle')]*5
pca = [pca]
i = 0
for train_index, test_index in kf.split(train_data_feature):
    i += 1
    train_x = train_data_feature[train_index]
    train_y = train_data_label[train_index]
    valid_x = train_data_feature[test_index]
    valid_y = train_data_label[test_index]
    # print(train_x.shape)
    # a = train_x
    # pca = PCA(n_components='mle',svd_solver='full',tol = 1)
    # print(pca)
    # # pca.fit(a)/
    # # train_x_pca = pca.transform(train_x)
    # # valid_x_pca = pca.transform(valid_x)
    # i = step - 1
    # pca_list[i].fit(train_x)
    # # train_x_pca, valid_x_pca = My_PCA(train_x, valid_x)
    # train_x_pca = pca_list[i].transform(train_x)
    # valid_x_pca = pca_list[i].transform(valid_x)   
    predict, fpr, tpr = svmpredict(train_x, train_y, valid_x, valid_y, c = C, k = 'rbf')
    AUC = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(i, AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with PCA(dimension == {}'.format(num))
    plt.legend(loc="lower right")
    print('\n****第{}折交叉验证, 惩罚系数为{}****'.format(i,C))
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

# %%
