# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 13:28:27 2020

@author: admin
"""
# 加载所需要的库
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
# 解决中文显示问题
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#%% 读取数据
ori_data = pd.read_excel('作业数据_2020合成.xls')
# 获取男性女性数据
male_data = ori_data.loc[ori_data['性别 男1女0'] == 1]
female_data = ori_data.loc[ori_data['性别 男1女0'] == 0]
# male_data.head() # 观察男性数据
#%% 第一题
# 获取男性女性50m数据 并删除nan数据
# male_50 = male_data[male_data['50米成绩'].notnull()]['50米成绩']
# female_50 = female_data[female_data['50米成绩'].notnull()]['50米成绩']
male_50 = male_data['50米成绩']
male_50.dropna(inplace = True)
female_50 = female_data['50米成绩']
female_50.dropna(inplace = True)
# 画图
plt.figure()
plt.hist(x = male_50, color='blue')
plt.hist(x = female_50, color='red')
plt.xlabel('50米成绩')
plt.ylabel('数量')
plt.title('男性女性50米成绩直方图')
plt.legend(['男性','女性'])

#%% 第二题 最大似然估计方法
# 身高
print('*********************\n')
print('最大似然结果:')
male_height = male_data['身高(cm)'] # 处理身高
female_height = female_data['身高(cm)']
male_height.dropna() # 删除nan
female_height.dropna()
[male_u_height, male_sig_height] = norm.fit(male_height) # loc = data.mean() scale = np.sqrt(((data - loc)**2).mean())
[female_u_height, female_sig_height] = norm.fit(female_height)
print('身高的参数:男性均值{},方差{},女性均值{},方差{}'.format(male_u_height, male_sig_height, female_u_height, female_sig_height))
# 体重
male_weight = male_data['体重(kg)'] # 体重身高
female_weight = female_data['体重(kg)']
male_weight.dropna() # 删除nan
female_weight.dropna()
[male_u_weight, male_sig_weight] = norm.fit(male_weight) 
[female_u_weight, female_sig_weight] = norm.fit(female_weight)
print('体重的参数:男性均值{},方差{},女性均值{},方差{}'.format(male_u_weight, male_sig_weight, female_u_weight, female_sig_weight))
# 50米
[male_u_50, male_sig_50] = norm.fit(male_50) 
[female_u_50, female_sig_50] = norm.fit(female_50)
print('50米参数:男性均值{},方差{},女性均值{},方差{}'.format(male_u_50, male_sig_50, female_u_50, female_sig_50))

#%% 第三题 贝叶斯估计方法 参数均值服从N(0,1), 方差根据第二题解得
# 身高
print('\n*********************\n')
print('贝叶斯估计结果:')
u0 = 0
sig0 = 1
male_N = len(male_height)
male_u_height_bayes = (1/(male_N + male_sig_height**2))*(male_height.sum())
female_N = len(female_height)
female_u_height_bayes = (1/(female_N + female_sig_height**2))*(female_height.sum())
print('身高的参数:男性均值{},女性均值{}'.format(male_u_height_bayes, female_u_height_bayes))
# 体重
male_N = len(male_weight)
male_u_weight_bayes = (1/(male_N + male_sig_weight**2))*(male_weight.sum())
female_N = len(female_weight)
female_u_weight_bayes = (1/(female_N + female_sig_weight**2))*(female_weight.sum())
print('体重的参数:男性均值{},女性均值{}'.format(male_u_weight_bayes, female_u_weight_bayes))
print('\n*********************\n')
#%% 第四题 求决策面
male = male_data[['身高(cm)','体重(kg)']]
male = np.array(male)
female = female_data[['身高(cm)','体重(kg)']]
female = np.array(female)
plt.scatter(male[:,0], male[:,1], alpha = 0.6)
plt.scatter(female[:,0], female[:,1], alpha = 0.6)

plt.xlabel('身高')
plt.ylabel('体重')
# 先验概率
P_male = len(male)/(len(male) + len(female))
P_famale = 1 - P_male
# 协方差矩阵
sig_male = np.cov(male.T)
sig_female = np.cov(female.T)
# 均值
mean_male = np.array([male_u_height, male_u_weight]).reshape(-1,1) # 列向量
mean_female = np.array([female_u_height, female_u_weight]).reshape(-1,1)
# 构建决策面
sample_height = np.linspace(150,200,50) # 构建50*50的一个待检测区域
sample_weight = np.linspace(40,100,50)
sample = np.zeros((50, 50))
for i in range(50):
    for j in range(50):
        x = np.array([sample_height[i],sample_weight[j]]).reshape(-1,1)
        sample[i,j] = 0.5 * (np.dot(np.dot((x-mean_male).T,np.linalg.inv(sig_male)), (x-mean_male))-\
        np.dot(np.dot((x-mean_female).T,np.linalg.inv(sig_female)), (x-mean_female))) +\
        0.5 * math.log(np.linalg.det(sig_male)/np.linalg.det(sig_female)) - math.log(P_male/P_famale)

plt.contour(sample_height, sample_weight, sample, 0, colors = 'green',linewidths=2)
# 画待区分的点
plt.scatter(170, 52, norm = 2, c = 'red', marker='s')
plt.scatter(178, 71, norm = 2, c = 'red', marker='s')

plt.legend(['男性','女性','待检测'])

# %%
