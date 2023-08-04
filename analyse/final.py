import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# import xgboost as xgb
from pandas.plotting import scatter_matrix
from sklearn import *
from sklearn.manifold import TSNE
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV,learning_curve,train_test_split,validation_curve
from matplotlib import cm
import seaborn as sns
from scipy import stats
import math

def randomforest(x_train, x_test, y_train, y_test):
    param_grid = {
        # 'bootstrap': [True],
        # 'max_depth': [6, 8, 10, 12],
        # 'max_features': [20, 30, 40, 50],
        # 'min_samples_leaf': [5, 10, 15],
        # 'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300, 1000]
    }

    modelrandom = RandomForestClassifier(oob_score=True,class_weight='balanced')

    clf = GridSearchCV(estimator=modelrandom, param_grid=param_grid, scoring='accuracy', cv=10)  # 5折交叉验证
    clf.fit(x_train, y_train)

    predictionsrandom = clf.predict(x_test)

    from sklearn.metrics import classification_report
    print(classification_report(y_test, predictionsrandom))
    print(accuracy_score(y_test,predictionsrandom))
    return predictionsrandom

def score(data):
    # data = pd.read_csv('food.csv')
    weight = pd.read_csv('weight.csv')
    # array = data.values
    we = weight.values
    # X = array[:, :-1]
    X = data
    n1, n2 = X.shape
    temp = []
    datamatrix = np.zeros((1, n2))
    # 自定义了数据来源的权重，共8维：wewe=8*1
    infer = we[0:2,]
    feacal = we[2:4,]
    type1 = we[4:6,]
    type2 = we[6:8,]

    weightpos = np.array([1,0])
    weightneg = np.array([0,1])


    we1 = np.dot(infer.T, weightneg)
    we2 = np.dot(feacal.T, weightneg)
    we3 = np.dot(type1.T, weightneg)
    we4 = np.dot(type2.T, weightpos)

    u1 = np.dot(X,we1)
    u2 = np.dot(X,we2)
    u3 = np.dot(X,we3)
    u4 = np.dot(X,we4)
    dic = {}
    # for k in range(len(u1)):
    #     if u1[k] in dic.keys():
    #         dic[u1[k]] = dic[u1[k]] + 1
    #     else:
    #         dic[u1[k]] = 1
    # print(dic)
    # print(u1.shape)
    # print(u2.shape)
    # print(u3.shape)
    # print(u4.shape)
    tru_sim = np.zeros((n1, n1))
    X = np.transpose(X)
    for i in range(0,n1):
        tru_sim[i,i]=0
        for j in range(i+1,n1):
            tru_sim[i,j] = user_similarity_on_modified_cosine(X[:,i],X[:,j])
            tru_sim[j,i] = user_similarity_on_modified_cosine(X[:,i],X[:,j])
        print("i = ",i)
    # print("distance:",tru_sim)
    # np.savetxt("acs04.txt", tru_sim)
    # tru_sim = metrics.pairwise_distances(X,metric="euclidean")
    # print("cosine:",D)
    p1 = np.dot(u1.T,tru_sim)
    p2 = np.dot(u2.T,tru_sim)
    p3 = np.dot(u3.T,tru_sim)
    p4 = np.dot(u4.T,tru_sim)
    for p in [p1,p2,p3,p4]:
        for i in range(len(p)):
            p[i] = p[i] / np.sum(tru_sim[:,i])
    # df1 = pd.DataFrame(p1, columns=['infer'])
    # df2 = pd.DataFrame(p2, columns=['feacal'])
    # df3 = pd.DataFrame(p3, columns=['type1'])
    # df4 = pd.DataFrame(p4, columns=['type2'])
    df = pd.DataFrame({'infer':p1,'faecal':p2,'type1':p3,'type4':p4})
    # df = pd.concat([df1, df2, df3, df4], sort=False)
    # print(df)
    df.to_csv("acs04.csv")


def user_similarity_on_modified_cosine(x,y):
    common = []
    for i in range(len(x)):
        if x[i] >0 and y[i]>0:
            common.append(i)
    if len(common) == 0:  # no common item of the two users
        return 0
    average1 = float(sum(x)) / len(x)
    average2 = float(sum(y)) / len(y)
    # denominator
    multiply_sum = sum((x[j] - average1) * (y[j] - average2) for j in common)
    # member
    pow_sum_1 = sum(math.pow(x[m] - average1, 2) for m in range(len(x)))
    pow_sum_2 = sum(math.pow(y[n] - average2, 2) for n in range(len(y)))

    modified_cosine_similarity = float(multiply_sum) / math.sqrt(pow_sum_1 * pow_sum_2)
    return modified_cosine_similarity

def jaccard_similarity(x,y):
    intersection_cardinality = len(set.intersection(*[set(x),set(y)]))
    union_cardinality = len(set.union(*[set(x),set(y)]))
    return intersection_cardinality/float(union_cardinality)


def cosine_dis(x,y):
    num = sum(map(float,x*y))
    denom = np.linalg.norm(x)*np.linalg.norm(y)
    return round(num/float(denom),3)
def cal_results():
    # df = pd.read_csv('jaccard01.csv')
    df = pd.read_csv('acs04.csv')
    p1 = df['infer']
    p2 = df['faecal']
    p3 = df['type1']
    p4 = df['type4']
    px = [p1, p2, p3, p4]
    k=0
    for p in [p1,p2,p3,p4]:
        p1_sorted = sorted(p,reverse=True)
        p1_id = []
        for i in range(len(p1_sorted)):
            x = p1_sorted.index(p[i])
            p1_id.append(x)
        px[k] = p1_id
        k =k +1
    # print(px[0])
    # print(p1)
    k = 100
    p1_name,p1_food,p1_y = find_name(px[0][:k])
    p2_name,p2_food,p2_y = find_name(px[1][:k])
    p3_name,p3_food,p3_y = find_name(px[2][:k])
    p4_name,p4_food,p4_y = find_name(px[3][:k])
    p = np.concatenate((p1_food.T,p2_food.T,p3_food.T,p4_food.T))
    # p = (p1_food.T)
    # print(p.shape)
    # print(p1_food.shape)
    pna = p1_name+p2_name+p3_name+p4_name
    y_class = p1_y+p2_y+p3_y+p4_y
    # y_class = p1_y
    tran = tsne(p,y_class)
    # tsne_class(tran,y_class)

    records = []
    for m in range(len(p1)):
        records.append([0,m,p2[m],p1[m]])
    print(records)
    print(RMSE(records))
    print(MAE(records))
    records = []
    for m in range(len(p1)):
        records.append([1, m, p3[m], p1[m]])
    print(RMSE(records))
    print(MAE(records))
    records = []
    for m in range(len(p1)):
        records.append([2, m, p4[m], p1[m]])
    print(RMSE(records))
    print(MAE(records))
    #
    # ap0 = AP(px[0],px[1])
    # ap1 = AP(px[0],px[2])
    # ap2 = AP(px[0],px[3])
    # # print(px[0],px[1])
    # # print(px[0],px[2])
    #
    # print(ap0,ap1,ap2)
    # s1 = stats.spearmanr(p1,p2)
    # s2 = stats.spearmanr(p1,p3)
    # s3 = stats.spearmanr(p1,p4)
    # print(s1,s2,s3)
    # inter = hit(px[2][:100],px[0][:100])
    # print(inter)
def tsne(matrix,y):
    model = TSNE(learning_rate=100,n_components=2,init='pca')
    transformed = model.fit_transform(matrix)
    # transformed = PCA(2).fit_transform(matrix)
    color = ['r', 'g', 'b', 'c']
    clabel = ['Inference','Fecal','MENDA1','MENDA2']
    target = [1.0, 15.0, 9.0, 11.0, 12.0, 16.0, 20.0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0, 13.0, 18.0, 19.0, 25.0]
    target_name = ['Dairy and Egg Products', 'Finfish and Shellfish Products', 'Fruits and Fruit Juices',
                   'Vegetables and Vegetable Products',
                   'Nut and Seed Products', 'Legumes and Legume Products', 'Cereal Grains and Pasta',
                   'Spices and Herbs', 'Fats and Oils', 'Poultry Products', 'Soups, Sauces, and Gravies',
                   'Sausages and Luncheon Meats', 'Pork Products', 'Beef Products',
                   'Baked Products', 'Sweets', 'Restaurant Foods']
    # print(transformed)
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }
    font2 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    plt.figure(figsize=(12,8))
    n1 = 100

    # plt.subplot(1, 2, 1)
    # for i in range(4):
    #     plt.scatter(transformed[n1*i:n1*(i+1), 0], transformed[n1*i:n1*(i+1), 1],c=color[i], label=clabel[i])
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,prop = font1)
    # s1 = 'Top 100 in four resources'
    # plt.title(s1,font2)
    #
    # plt.subplot(1, 2, 2)

    p = []

    for i in range(len(target)):
        s1 = 0
        ccolor = cm.rainbow(int(255 / 17) * (i + 1))
        # plt.scatter(transformed[y == target[i], 0], transformed[y == target[i], 1], c=ccolor, label=target_name[i])
        for j in range(n1):
            if y[j] == target[i]:
                plt.scatter(transformed[j, 0], transformed[j, 1], c=ccolor)
        plt.scatter([], [], c=ccolor, label=target_name[i])
    # for i in range(len(target)):
    #     ccolor = cm.rainbow(int(255 / 17) * (i + 1))
    #     plt.scatter([],[],c=ccolor,label=target_name[i])
    s2 = 'Categories of top 100 food based on inference'
    plt.title(s2,font2)
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,prop = font1)
    plt.show()

    return transformed

def tsne_class(transformed,y):
    color = ['grey', 'gold', 'darkviolet', 'turquoise', 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'darkorange', 'lightgreen',
             'plum', 'tan', 'khaki', 'pink']
    target = [1.0, 15.0, 9.0, 11.0, 12.0, 16.0, 20.0, 2.0, 4.0, 5.0, 6.0, 7.0, 10.0, 13.0, 18.0, 19.0, 25.0]
    target_name = ['Dairy and Egg Products', 'Finfish and Shellfish Products', 'Fruits and Fruit Juices',
                   'Vegetables and Vegetable Products',
                   'Nut and Seed Products', 'Legumes and Legume Products', 'Cereal Grains and Pasta',
                   'Spices and Herbs', 'Fats and Oils', 'Poultry Products', 'Soups, Sauces, and Gravies',
                   'Sausages and Luncheon Meats', 'Pork Products', 'Beef Products',
                   'Baked Products', 'Sweets', 'Restaurant Foods']
    # print(transformed.shape)
    # print(y)
    plt.figure(figsize=(10, 6))
    p = []
    for i in range(len(target)):
        ccolor = cm.rainbow(int(255 / 17) * (i + 1))
        temp = []
        s1 = 0
        for j in range(len(y)):
            if y[j] == target[i]:
                plt.scatter(transformed[j,0], transformed[j,1], c=ccolor)
        # print(temp)

    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,labels=target_name)
    plt.show()
def find_name(p):
    fname = []
    fdf = pd.read_csv('foodname.csv')
    fid = fdf['fdc_id']
    array = fdf.values
    X = array[:, 2:-1]
    Y = fdf['type']
    n = len(p)
    datamatrix = np.zeros((85, n))
    fn = fdf['foodname']
    yc = []
    for i in range(n):
        fname.append(fn[p[i]])
        datamatrix[:,i] = X[p[i],:]
        yc.append(Y[p[i]])
    # print(yc)
    return fname,datamatrix,yc
    # print(fname)
    # print(datamatrix)
def AP(ranked_list, ground_truth):
    """Compute the average precision (AP) of a list of ranked items
    """
    hits = 0
    sum_precs = 0
    for n in range(len(ranked_list)):
        if ranked_list[n] in ground_truth:
            hits += 1
            sum_precs += hits / (n + 1.0)
    if hits > 0:
        return sum_precs / len(ground_truth)
    else:
        return 0


def hit(gt_items, pred_items):
    count = 0
    for item in pred_items:
        if item in gt_items:
            count += 1
    return count

# 定义精确率指标计算方式
def precision(self):
    all, hit = 0, 0
    for user in self.test:
        test_items = set([x[0] for x in self.test[user]])
        rank = self.recs[user]
        for item, score in rank:
            if item in test_items:
                hit += 1
        all += len(rank)
    return round(hit / all * 100, 2) if all > 0 else 0.0

    # 定义召回率指标计算方式
def recall(self):
    all, hit = 0, 0
    for user in self.test:
        test_items = set([x[0] for x in self.test[user]])
        rank = self.recs[user]
        for item, score in rank:
            if item in test_items:
                hit += 1
        all += len(test_items)
    return round(hit / all * 100, 2) if all > 0 else 0.0

def draw_heatmap(px):
    s12 = cosine_dis(px[0],px[1])
    s13 = cosine_dis(px[0],px[2])
    s14 = cosine_dis(px[0],px[3])
    s23 = cosine_dis(px[1],px[2])
    s24 = cosine_dis(px[1],px[3])
    s34 = cosine_dis(px[2],px[3])
    dfcorr = [s12,s13,s14,s23,s24,s34]
    ylabels = ['infer', 'faecal', 'type1', 'type2']
    dfData = df.corr()
    for i in range(4):
        dfData.values[i][3-i] = dfcorr[i]
        dfData.values[3 - i][i] = dfcorr[i]
    print(dfData)
    # # print(df)
    plt.subplots(figsize=(15, 15))  # 设置画面大小
    sns.heatmap(dfData, square=True, yticklabels=ylabels, xticklabels=ylabels, cmap="YlGnBu")
    plt.show()

def RMSE(records):
    return math.sqrt(sum([(rui - pui) * (rui - pui) for u, i, rui, pui in records]) / float(len(records)))


def MAE(records):
    return sum([abs(rui - pui) for u, i, rui, pui in records]) / float(len(records))

def draw(data):
    data.hist(figsize=(16,14))
    plt.show()

def normalization(X):
    n1,n2 = X.shape
    datamatrix = np.zeros((1, n2))

    temp = np.zeros((1, n2))
    for x in range(n1):
        rows = X[x]
        # count = np.count_nonzero(rows)
        # if count < 3:
        #     continue
        minVals = min(rows)

        maxVals = max(rows)

        # count = np.count_nonzero(rows)
        ranges = maxVals - minVals
        if ranges == 0:
            continue
        ranges = maxVals - minVals
        for i in range(n2):
            rows[i] = (rows[i] - minVals)/ranges
        temp[0,:] = rows
        datamatrix = np.concatenate([datamatrix,temp])

    datamatrix = datamatrix[1:,:]

    # print(datamatrix.shape)
    # print(datamatrix[5,:])
    # print(title.shape)
    return datamatrix

def draw_results():
    name_list = ['Inference+Fecal', 'Inference+MENDA1', 'Inference+MENDA2']
    num_list = [0.74, 1.92, 1.12]
    num_list1 = [0.53, 1.53, 0.91]
    x = list(range(len(num_list)))
    # total_width, n = 0.8, 2
    # width = total_width / n
    plt.figure(figsize=(12, 8),dpi=80)
    width = 0.3
    plt.bar(np.arange(len(num_list)), num_list, width=width,label='RMSE',tick_label=name_list , fc='#9999ff')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    plt.bar(np.arange(len(num_list1))+width, num_list1, width=width, label='MAE', fc='#ff9999')
    index = np.arange(len(name_list))
    for a, b in zip(index, num_list):  # 柱子上的数字显示
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=12)
    for a, b in zip(index + width, num_list1):
        plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=12)

    plt.tick_params(axis='both', labelsize=18 )
    font2 = {'family': 'Times New Roman',
             'weight': 'bold',
             'size': 20,
             }
    s = 'Results of nutrition recommendation'
    plt.title(s,font2)

    plt.legend(fontsize=18)
    plt.show()

if __name__ == '__main__':
    # score()


    # data = pd.read_csv('food.csv')
    # array = data.values
    # X = array[:, :-1]
    # data = normalization(X)
    # score(data)

    # cal_results()
    draw_results()
    # x = pd.DataFrame(X.T)
    # draw(x)
    # # 根据样本和新标签Y 划分训练集测试集
    # x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y,test_size=0.2,random_state=42,stratify=Y)
    # predrandom = randomforest(x_train, x_test, y_train, y_test)