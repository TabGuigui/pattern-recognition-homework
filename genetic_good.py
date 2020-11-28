'''
author : tabgui
time : 2020.11.12
content : 遗传算法实现男女数据集特征选择
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
from sklearn.svm import SVC
from svm import svmpredict
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ori_data = pd.read_excel('作业数据_2020合成.xls')
from pr_config import My_count, My_cv_iterator, My_bp_network,My_CV
np.random.seed(100)
random.seed(1)
# %% 简单观察数据
used_feature = ['性别 男1女0','身高(cm)','体重(kg)','鞋码','50米成绩','肺活量','喜欢运动']
 
train_data = ori_data[used_feature]
train_data["身高(cm)"]=preprocessing.scale(train_data["身高(cm)"]) 
train_data["体重(kg)"]=preprocessing.scale(train_data["体重(kg)"]) 
train_data["50米成绩"]=preprocessing.scale(train_data["50米成绩"]) 
train_data["鞋码"]=preprocessing.scale(train_data["鞋码"]) 
train_data['肺活量'] = preprocessing.scale(train_data['肺活量'])
train_data.head()
train_data.describe()
train_data.info()
train_data.dropna(inplace = True)
# %%
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
# %%
def Feature_choose(feature, code):
    '''
    input
    feature : 待特征选择的特征矩阵 m*n array m samplenum n featurenum
    code : 编码 n维list 某一位为1代表选中

    '''
    index = [i for i,x in enumerate(code) if x == 1]
    feature = feature[:,index]
    return feature
def Divergence_compute(feature,label):
    '''
    计算准则函数 这里采用散度
    input
    feature : 特征选择后的特征
    label : 一维标签向量
    '''
    index_male = np.where(label == 1)
    index_female = np.where(label == 0)
    feature_male = feature[index_male]
    feature_female = feature[index_female]
    male_mean = np.mean(feature_male, axis = 0)
    female_mean = np.mean(feature_female, axis = 0)   
    male_cov = np.cov(feature_male.T)
    female_cov = np.cov(feature_female.T)
    if female_cov.shape == ():
        J = 1/2*(1/male_cov*female_cov + 1/female_cov*male_cov-2)+\
            1/2*np.dot((male_mean - female_mean).T,(1/male_cov+1/female_cov)).dot((male_mean-female_mean))
    else:
        J = 1/2*np.trace(np.linalg.inv(male_cov).dot(female_cov)+np.linalg.inv(female_cov).dot(male_cov)-2*np.identity(male_cov.shape[0]))+\
            1/2*np.dot((male_mean - female_mean).T,(np.linalg.inv(male_cov)+np.linalg.inv(female_cov))).dot((male_mean-female_mean))
    return J
def Fitness_compute(x):
    index = []
    for i in range(len(x)):
        if x[i] == 1:
            index.append(i)
    #男女生样本集
    boy_data = male_data[:,[index]]
    girl_data = female_data[:,[index]]
    boy_mean = np.mean(boy_data, axis=0)  # 男生各项均值
    girl_mean = np.mean(girl_data, axis=0)  # 女生各项均值
    mean = boy_mean * (len(boy_data) / (len(train_data))) + girl_mean * (len(girl_data) / len(train_data))  # 总体均值

    # 男生类内离差矩阵
    boy_Sw = np.zeros(shape=(len(index), len(index)))
    for i in range(len(boy_data)):
        boy_Sw += (boy_data[i] - boy_mean).reshape(-1, 1) * (boy_data[i] - boy_mean)
    boy_Sw = boy_Sw / len(boy_data)
    # 女生类内离差矩阵
    girl_Sw = np.zeros(shape=(len(index), len(index)))
    for i in range(len(girl_data)):
        girl_Sw += (girl_data[i] - girl_mean).reshape(-1, 1) * (girl_data[i] - girl_mean)
    girl_Sw = girl_Sw / len(girl_data)
    # 类内离差
    Sw = boy_Sw * (len(boy_data) / len(train_data)) + girl_Sw * (len(girl_data) / len(train_data))
    # 类间离差
    Sb = (boy_mean - mean).reshape(-1, 1) * (boy_mean - mean) * (len(boy_data) / len(train_data)) \
         + (girl_mean - mean).reshape(-1, 1) * (girl_mean - mean) * (len(girl_data) / len(train_data))
#     J = (np.trace(Sb))/(np.trace(Sw))
    J=np.trace(Sb)/np.trace(Sw)
    return J
# %%
class My_genetic():
    def __init__(self, cross_rate = 1, variate_rate = 0.8,num_population=10,num_feature=6):
        self.cross_rate = cross_rate
        self.variate_rate = variate_rate
        self.num_population = num_population
        self.num_feature = num_feature
        self.j_max_ite = [] # 每轮最大适应度
        self.Jmax = 0 # 全局最大适应度
        self.best_feature = [] # 最好的特征
    def init(self):
        # 生成初始种群
        origin_population = np.random.randint(0,2,size = (self.num_population,self.num_feature))
        return origin_population
    def divergence_compute(self, population):
        j_list = []
        for i in range(self.num_population): # 对每一个编码进行散度计算
            feature = Feature_choose(train_data_feature,population[i])
            J = Divergence_compute(feature,train_data_label)
            j_list.append(J)
        j_max = max(j_list)
        self.j_max_ite.append(j_max)
        if j_max > self.Jmax:
            self.Jmax = j_max
            self.best_feature = population.copy()[j_list.index(j_max)]
        return j_list
    def fitness_compute(self, population):
        j_list = []
        for i in range(self.num_population):
            J = Fitness_compute(population[i])
            if np.isnan(J) == 1: # array中处理nan 需要注意
                j_list.append(0)
            else:
                j_list.append(J)
        j_max = max(j_list)
        self.j_max_ite.append(j_max)
        if j_max > self.Jmax:
            self.Jmax = j_max
            self.best_feature = population[j_list.index(j_max)]
            print('全局最好修改为',population[j_list.index(j_max)])
            print('全局最好是硬度维:',self.Jmax)
        else:
            print('不修改')
        # print('***现在最好的是****',self.best_feature)
        return j_list
    def prob_comput(self, j_list):
        p = [x/(sum(j_list)) for x in j_list]
        prob_list = [] # 求概率列表
        b = 0
        for i in range(self.num_population):
            b += p[i]
            prob_list.append(b)
        return prob_list
    def update(self, prob_list, population):
        new_population = []
        for i in range(self.num_population):
            prob = np.random.random()
            if prob < prob_list[0]:
                new_population.append(population[0])
            else:
                for j in range(len(prob_list)):
                    if prob >= prob_list[j] and prob < prob_list[j+1]:
                        new_population.append(population[j+1])
        return new_population
    # def cross(self, population):
    #     a = random.random()
    #     new_population = population.copy()
    #     if a < self.cross_rate: # 进行交叉工作
    #         # np.random.shuffle(new_population)
    #         cpoint=random.randint(0,self.num_population)
    #         for i in range(0,self.num_population,2):
    #             b = new_population[i].copy()[:cpoint]
    #             c = new_population[i+1].copy()[:cpoint]
    #             new_population[i][:cpoint] = c
    #             new_population[i+1][:cpoint] = b
    #     return new_population
    def cross(self,population):
    # 一定概率杂交，主要是杂交种群种相邻的两个个体
        newPopulation = population
        for i in range(self.num_population-1):
            if(random.random()<self.cross_rate):
                cpoint=random.randint(0,5 )#单点交叉
                temporary1=[]
                temporary2=[]
                #第i和i+1个染色体交叉
                temporary1.extend(population[i][0:cpoint])
                temporary1.extend(population[i+1][cpoint:len(population[i])])
                temporary2.extend(population[i+1][0:cpoint])
                temporary2.extend(population[i][cpoint:len(population[i])])
                newPopulation[i]=temporary1
                newPopulation[i+1]=temporary2
        return newPopulation
    def variate(self, population):
        a = random.random()
        if a > self.variate_rate: # 进行变异工作
            print('****变异****')
            for popu in population:
                point = random.randint(0,self.num_feature-1)
                if popu[point] == 1:
                    popu[point] = 0
                else:
                    popu[point] = 1
        return population
# def svmpredict(train_data,train_label,test_data,test_label, c,k = 'rbf', g = 'auto'):
#     '''
#     input
#     c ： 惩罚系数
#     k :  核函数 默认高斯核
#     output
#     predict : 预测结果
#     ''' 
#     svc = SVC(kernel=k,C=c,gamma = g, probability= True)
#     svc = svc.fit(train_data,train_label)
#     predict_label = svc.predict(test_data)
#     predict_prob = svc.decision_function(test_data)
#     fpr,tpr,_ = roc_curve(test_label,predict_prob)
#     return predict_label,fpr,tpr
# %%
def main_genetic(iteration = 500, decision_function = 'fitness', variate = 0.9, population = 8):
    Genetic = My_genetic(cross_rate = 1, variate_rate = variate,num_population=population,num_feature=6)
    population = Genetic.init()
    j_max_list = []
    for i in range(iteration):
        print('进行第{}迭代'.format(i))
        if decision_function == 'fitness':
            j_list = Genetic.fitness_compute(population)
            prob_list = Genetic.prob_comput(j_list)
            # print('1',Genetic.best_feature)
            population = Genetic.update(prob_list, population)
            # print('1',Genetic.best_feature)
            population = Genetic.cross(population)
            # print('1.......',Genetic.best_feature)
            population = Genetic.variate(population)
            # print('1',Genetic.best_feature)
            j_max_list.append(Genetic.Jmax)
        elif decision_function == 'divergence':
            j_list = Genetic.divergence_compute(population)
            # print(population)
            prob_list = Genetic.prob_comput(j_list)
            print('1',Genetic.best_feature)
            population = Genetic.update(prob_list, population)
            print('2',Genetic.best_feature)
            population = Genetic.cross(population)
            print('2',Genetic.best_feature)
            population = Genetic.variate(population)
            print('2',Genetic.best_feature)
            j_max_list.append(Genetic.Jmax)
        else:
            raise ValueError('wrong parameter')
    plt.figure()
    plt.plot(np.array(range(iteration)),j_max_list)
    plt.plot(np.array(range(iteration)),Genetic.j_max_ite)
    plt.xlabel('迭代次数')
    plt.ylabel('适应度')
    plt.legend(('全局最大适应度','当前迭代次数最大适应度'))
    plt.show()
    print('最大适应度为:',Genetic.Jmax)
    print('最好的特征选择是:', Genetic.best_feature)
    return Genetic.best_feature
# %%
# best_feature = main(200,'fitness',0.9,8)
# feature_choose = Feature_choose(train_data_feature,best_feature)
# %%
# if __name__ == '__main__':
# best_feature = main(200,'fitness',0.9,8)
# feature_choose = Feature_choose(train_data_feature,best_feature)

best_feature = main_genetic(200,'fitness',0.9,8)
feature_choose = Feature_choose(train_data_feature,best_feature)
fold_num = 5
from sklearn.model_selection import KFold
kf = KFold(n_splits=fold_num)
se_mean = 0
sp_mean = 0
acc_mean = 0
C = 50
i = 0
plt.figure()
for train_index, test_index in kf.split(feature_choose):
    i += 1
    train_x = feature_choose[train_index]
    train_y = train_data_label[train_index]
    valid_x = feature_choose[test_index]
    valid_y = train_data_label[test_index]
    print(train_x.shape)
    print(train_y.shape)
    predict, fpr, tpr = svmpredict(train_x, train_y, valid_x, valid_y, c = C, k = 'rbf')
    AUC = auc(fpr, tpr)
    plt.plot(fpr,tpr,marker = 'o',label='{} fold ROC curve (area = {:.2f})' .format(i,AUC))
    plt.plot([0, 1], [0, 1], color='navy', lw=5, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic with C = {}'.format(C))
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
plt.show()
sp_mean = sp_mean/fold_num
se_mean = se_mean/fold_num
acc_mean = acc_mean/fold_num
print('\n平均SP为{}'.format(sp_mean))
print('\n平均SE为{}'.format(se_mean))
print('\n平均ACC为{}'.format(acc_mean))

# %%
