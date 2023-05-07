import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
import BOA as BOA
import warnings
from OriginalVibrationSignal import ovs_preprocess

warnings.filterwarnings("ignore")
'''优化函数'''
def fun(X):
    classifier=svm.SVC(C=X[0], kernel='rbf', gamma=X[1])
    classifier.fit(x_train, y_train)
    tes_label=classifier.predict(x_test) #测试集的预测标签
    train_labelout = classifier.predict(x_train) #测试集的预测标签
    output = 2 - accuracy_score(y_test, tes_label) - accuracy_score(y_train,train_labelout)#计算错误率，如果错误率越小，结果越优
    return output

num_classes = 10    # 样本类别
length = 784        # 样本长度
number = 200  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.5, 0.25, 0.25]  # 测试集验证集划分比例

path = r'data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = ovs_preprocess.prepro(
    d_path=path,
    length=length,
    number=number,
    normal=normal,
    rate=rate,
    enc=False, enc_step=28)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)

#设置蝴蝶参数
pop = 20 # 种群数量
MaxIter = 5 # 最大迭代次数
dim = 2 # 维度
lb = np.matrix([[0.001],[0.001], [0.001], [0.001]]) # 下边界
ub = np.matrix([[100],[100], [100], [100]])# 上边界
fobj = fun
GbestScore,GbestPositon, Curve = BOA.BOA(pop,dim,lb,ub,MaxIter ,fobj)
print('最优适应度值：',GbestScore)
print('c,g最优解：',GbestPositon)
#利用最终优化的结果计算分类正确率等信息
#训练svm分类器

classifier=svm.SVC(C=GbestPositon[0,0],kernel='rbf',gamma=GbestPositon[0,1]) # ovr:一对多策略
classifier.fit(x_train,y_train.ravel()) #ravel函数在降维时默认是行序优先
#4.计算svc分类器的准确率
tra_label=classifier.predict(x_train) #训练集的预测标签
tes_label=classifier.predict(x_test) #测试集的预测标签
print("训练集准确率：", accuracy_score(y_train,tra_label) )
print("测试集准确率：", accuracy_score(y_test,tes_label) )

#绘制适应度曲线
plt.figure(2)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('BOA',fontsize='large')
plt.show()








