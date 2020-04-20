from DataHelper import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import time

def train(method,data):
    if isinstance(data,list):
        data = np.concatenate(data,axis=0)
    x = data[:,:-1]
    y = data[:,-1]
    if method == 'KNN':
        clf = KNeighborsClassifier()
    elif method == 'SVM':
        clf = SVC(1.5,class_weight="balanced")
    elif method == 'Random Forest':
        clf = RandomForestClassifier()
    elif method == 'XGBoost':
        clf = XGBClassifier(max_depth=25)
    clf.fit(x,y)
    # print "training accuracy:%f"%(clf.score(x,y))
    return clf

def test(clf,data,criteria):
    # data = np.concatenate(data,axis=0)
    x = data[:,:-1]
    y = data[:,-1]
    if criteria == 'acc':
        result = clf.score(x,y)
        print ("test accuracy:%f"%result)
    if criteria == 'confusion matrix':
        y_pred = clf.predict(x)
        result = confusionMatrix(y_pred,y)
        # print result
    return result


def main():
    data = DataHelper('data/Twitter.csv')
    data.MinMaxScale()
    data = data.getdata()
    traindata = data[:4]
    # traindata = oversamplingSMOTE(traindata,read=True)
    traindata = np.concatenate(traindata)
    testdata = data[4:]
    # method = 'KNN'         # run abour 40s   acc 96.75%  f-measure 0.65
    # method = 'SVM'         # run about 30s   acc 95%     f-measure overfit(nan)
    # method = 'Random Forest' # run about 0.44s acc 98.28%  f-measure 0.79
    method = 'XGBoost'
    start = time.time()
    clf = train(method,traindata)
    acc = []
    imbalanced = []
    MDDT = [[5,9],[11,14],[16,18],[25,29],[31,37]]
    CUSUM = [[13,16],[23,24],[25,26],[28,30],[35,38]]
    PH = [[7,8],[13,14],[25,26]]
    for i in range(len(testdata)//4):
        print (i)
        last_i = 0
        testbatch = np.concatenate(testdata[i*4:(i+1)*4])
        try:
            acc.append(test(clf,testbatch,criteria = 'acc'))
            imbalanced.append(test(clf,testbatch,criteria='confusion matrix'))
        except:
            acc.append(0)
            imbalanced.append([0,0,0])
        testbatch = np.concatenate(testdata[i*4:(i+1)*4])
        clf.fit(testbatch[:,:-1],testbatch[:,-1])
        for t in MDDT:
            start = np.floor(t[0]/4.0) -1
            end = np.ceil(t[1]/4.0) -1
            if start == i:
                try:
                    testbatch = np.concatenate(testdata[t[0]-4:t[1]-4])
                    clf.fit(testbatch[:,:-1],testbatch[:,-1])
                except:
                    continue
                break
    df = pd.DataFrame(acc)
    # df.to_csv("MDDT_ACC_XGB.csv",header=None,index=None)
    df = pd.DataFrame(imbalanced)
    # df.to_csv("MDDT_CONFUSION_XGB.csv",header=None,index=None)
    print ("time:",time.time() - start)

if __name__ == '__main__':
    main()