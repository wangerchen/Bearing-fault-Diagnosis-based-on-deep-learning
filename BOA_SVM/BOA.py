import numpy as np
import random
import copy

# (1)计算适应度函数f(x), x = (x 1, ..., x d i m)
# (2)给每个蝴蝶生成ｎ个初始解xi = (i = 1, 2, ..., n)
# (3)声明变量c, α, g ∗, p
#
# (4)while未到终止条件do
#     (5)for每一个蝴蝶do
#          (6)采用式(1)计算其香味函数f
#     (7)end for
#
#     (8)找出最优的香味函数f，并赋值给g∗
#
#     (9)for 每一个蝴蝶do
#         (10)采用式(4)计算概率ｒ
#         (11) if r < p then
#             (12)采用式(2)进行全局搜索
#         (13) else
#             (14)采用式(3)进行局部随机搜索
#         (15)end if
#     (16)endfor
# (17)end while
# (18)输出最优解




''' 种群初始化函数 '''
def initial(pop, dim, ub, lb): # pop种群数量20， dim维度2
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]   # random.random() （0-1之间）

    return X, lb, ub


'''边界检查函数'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])    # 给一列全0
    for i in range(pop):
        fitness[i] = fun(X[i, :])   # fun是适应度，main中定义了，越小越好
    return fitness


'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)    # 适应度排序
    index = np.argsort(Fit, axis=0)   # 对应适应度的索引
    return fitness, index


'''根据适应度对位置进行排序'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def sensory_modality_NEW( x ,Ngen):
    y = x + (0.025 / (x * Ngen))
    return y


'''蝴蝶优化算法'''
def BOA(pop, dim, lb, ub, MaxIter, fun):
    
    p = 0.8 # probabibility switch
    power_exponent = 0.1
    sensory_modality = 0.01
    
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    indexBest = np.argmin(fitness)      # argmin显示最小值的下标
    GbestScore = fitness[indexBest]     # 将最小值给适应度
    GbestPositon = np.zeros([1, dim])    # 给一个一行两维的0值
    GbestPositon[0, :] = X[indexBest, :]   # X = np.zeros([pop, dim])
    X_new = copy.copy(X)
    Curve = np.zeros([MaxIter, 1])   # (50行，1列)全0
    for t in range(MaxIter):
        print("第"+str(t)+"次迭代")
        for i in range(pop):    # 种群数pop：20
            FP = sensory_modality * (fitness[i] ** power_exponent)
            
            if random.random() > p:
                dis = random.random() * random.random() * GbestPositon - X[i, :]
                Temp = np.matrix(dis*FP)
                X_new[i, :] = X[i, :] + Temp[0, :]
            else:
                # Find random butterflies in the neighbourhood
                # epsilon = random.random()
                Temp = range(pop)
                JK = random.sample(Temp, pop)   # 生成20个，1-20的随机数 [18, 2, 10, 13, 9, 12, 4, 0, 3, 1, 15, 5, 19, 17, 8, 7, 14, 16, 11, 6]
                dis = random.random() * random.random() * X[JK[0], :] - X[JK[1], :]
                Temp = np.matrix(dis*FP)
                X_new[i, :] = X[i, :] + Temp[0, :]
            for j in range(dim):
                if X_new[i,j] > ub[j]:
                    X_new[i, j] = ub[j]
                if X_new[i,j] < lb[j]:
                    X_new[i, j] = lb[j]
            
            # 如果更优才更新
            if (fun(X_new[i,:]) < fitness[i]):
                X[i, :] = copy.copy(X_new[i, :])
        
        X = BorderCheck(X, ub, lb, pop, dim)  # 边界检测
        fitness = CaculateFitness(X, fun)     # 计算适应度值
        indexBest = np.argmin(fitness)        # 找出适应度最小的
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])     # 得到适应度值
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore    # 绘制适应度曲线
        # 更新sensory_modality
        sensory_modality = sensory_modality_NEW(sensory_modality, t+1)

    return GbestScore, GbestPositon, Curve
