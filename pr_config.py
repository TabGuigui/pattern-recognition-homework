'''
第二次模式识别作业需要用到的相关函数与类
'''
import numpy as np
import random
def My_count(test_y , predict_y): # 结果统计
    '''
    input
    test_y : 测试集真实labebl 
    predict_y : 测试集预测label
    '''
    true_1_num = np.sum(test_y == 1)
    true_0_num = np.sum(test_y == 0)
    print('验证集类别为1（男生）的个数',true_1_num,'\n验证集类别为0（女生）的个数',true_0_num)
    
    predict_1_num = np.sum(predict_y == 1)
    predict_0_num = np.sum(predict_y == 0)
    print('预测为类别1的个数',predict_1_num,'\n预测为类别0的个数',predict_0_num)
    tp_matrix = test_y * predict_y # tp_matrix中为1的数据为真阳
    tp = np.sum(tp_matrix == 1)
    fp = predict_1_num - tp
    fn = true_1_num - tp
    tn = true_0_num - fp
    print('true positive number:',tp)
    print('false positive number:', fp)
    print('fasle negetive number:', fn)
    print('true negetive number:', tn)
    P_male = tp/(tp+fp)
    P_female = tn/(tn+fn)
    R_male = tp/(tp+fn) # 男性召回率 se
    R_female = tn/(tn+fp) # 女性召回率 sp
    print('男性查准率为: {}, 召回率为: {}'.format(P_male, R_male))
    print('女性查准率为: {}, 召回率为: {}'.format(P_female, R_female))
    ACC = (tp + tn)/(tp+fp+fn+tn)
    print('算法精确率(ACC)为：{}'.format(ACC))
    return R_male, R_female, ACC


def My_CV(train_feature, train_label, cv_number): 
    train = np.hstack((train_feature, train_label.reshape(-1,1)))
    m,n = train.shape
    ratio = 1/cv_number
    train_number = int(m * (1-ratio))
    random.shuffle(train)
    train_data = train[:train_number,:n-1]
    train_label = train[:train_number,-1]
    valid_data = train[train_number:,:n-1]
    valid_label = train[train_number:,-1]
    return train_data,train_label,valid_data,valid_label


class My_cv_iterator(): # 自写交叉验证
    def __init__(self,data,label,cv_num):
        '''
        data ：输入特征
        label : 输入标签
        cv_num : k-fold 折数

        return
        返回k-fold的iterator

        e.g
        for step, train, valid in My_cv_iterator()
        '''
        self.data = data
        self.label = label
        self.cv_num = cv_num
        train = np.hstack((self.data, self.label.reshape(-1,1)))
        m,n = train.shape
        ratio = 1/self.cv_num
        every_fold_num = int(m*ratio)
        self.train_fold = []
        self.valid_fold = []
        for i in range(self.cv_num):
            valid_data = train[every_fold_num*i:every_fold_num*(i+1)]
            aset = set([tuple(x) for x in train])
            bset = set([tuple(x) for x in valid_data])
            train_data = np.array([x for x in aset-bset])
            self.valid_fold.append(valid_data)
            self.train_fold.append(train_data)     
        self.cur = 0 # 迭代器当前迭代次数
    def __iter__(self):
        return self
    def __next__(self):
        if self.cur >= len(self.train_fold):
            raise StopIteration
        cur_train_data = self.train_fold[self.cur]
        cur_valid_data = self.valid_fold[self.cur]
        self.cur += 1
        return self.cur, cur_train_data,cur_valid_data

class My_bp_network():
    def __init__(self, input_num, hidden_layer, output_num):
        '''
        input

        input_num ：输入维度
        hidden_layer ：隐藏层节点数
        output_num ：输出维度
        
        '''
        self.hidden_layer = hidden_layer
        self.input_num = input_num
        self.output_num = output_num
        self.weight1 = np.random.uniform(-1,1,(self.hidden_layer,self.input_num))
        self.bias1 = np.zeros((1, self.hidden_layer))
        self.weight2 = np.random.uniform(-1,1,(self.output_num,self.hidden_layer))
        self.bias2 = np.zeros((1,1))
    def sigmoid(self,x):
        y = 1/(1+np.exp(-x))
        return y
    def forward(self,x):
        '''
        x : (m*1) m 为特征数
        '''
        self.input = x
        self.hidden_layer_out = np.dot(self.weight1,x) + self.bias1 # 隐藏层
        self.hidden_layer_out = self.sigmoid(self.hidden_layer_out) # 激活函数
        # print(hidden_layer_out)
        self.final_layer_out = np.dot(self.weight2,self.hidden_layer_out.T) + self.bias2 # 输出层
        # print(final_layer_out)
        output = self.sigmoid(self.final_layer_out)
        return output
    def backward(self, lr, yhat, y):
        '''
        lr ：学习率
        yhat : 前向传播输出
        y ：实际标签
        '''
        dLdy = -(y-yhat) # loss对yhat求导
        dydxfinal = yhat*(1-yhat) # yhat对 sum(w(2)x) 求导 yhat = sigmoid(xfinal) 
        dLdxfinal = dLdy * dydxfinal # loss 对 sum(w(2)x) 求导 
        # print('loss对输出层wx的和的梯度',dLdxfinal)
        self.bias2 -= lr*dLdxfinal # sum(w(2)x)对b求导 
        for i in range(self.hidden_layer): # 隐藏层从上到下梯度下降
            dxfinaldwi = self.hidden_layer_out[0][i] # sum(w(2)x)对w(2)i求导
            dxfinaldxi = self.weight2[0][i] # sum(w(2)x)对xi求导
            self.weight2[0][i] -= lr * dLdxfinal * dxfinaldwi # loss对w(2)i求导
            dxidhiddenout = self.hidden_layer_out[0][i]*(1-self.hidden_layer_out[0][i])
            self.bias1[0][i] -= lr * dLdxfinal * dxfinaldxi * dxidhiddenout
            for j in range(self.input_num): # 输入层从上到下梯度下降
                dhiddenoutdwij = self.input[j]
                # print(dxfinaldhiddenout*dhiddenoutdwij)
                self.weight1[i][j] -= lr*dLdxfinal * dxfinaldxi * dxidhiddenout * dhiddenoutdwij