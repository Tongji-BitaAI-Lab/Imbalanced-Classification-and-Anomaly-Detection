'-* author: xuesong wang *-'
import pandas as pd
import numpy as np
from random import sample
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.preprocessing import MinMaxScaler
# from imblearn.over_sampling import SVMSMOTE,SMOTE,ADASYN
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.animation import FuncAnimation
warnings.filterwarnings("ignore")


class DataHelper:
    def __init__(self,path):
        data = random_order(path)
        self.data = data

    def getdata(self):
        return self.data

    def MinMaxScale(self):
        scaler = MinMaxScaler()
        data_scaled = []
        data = self.data
        for t,batchdata in enumerate(data):
            if t == 0:
                x = scaler.fit_transform(batchdata.iloc[:,:-1])
            else:
                x = scaler.transform(batchdata.iloc[:,:-1])
            y = np.reshape(batchdata.iloc[:,-1].values,[-1,1])
            data_scaled.append(np.concatenate((x,y),axis=1))
        self.data = data_scaled
        return data_scaled


def oversamplingSMOTE(data,read=False):
    if read == False:
        data = np.concatenate(data)
        x = data[:,:-1]
        y = data[:,-1].astype(int)
        x_resampled, y_resampled = SVMSMOTE().fit_resample(x,y)
        data_resampled = np.concatenate([x_resampled,y_resampled.reshape([-1,1])],axis=1)
        df = pd.DataFrame(data_resampled)
        df.to_csv("data/resampled_data.csv",index=None,header=None)
    else:
        data_resampled = pd.read_csv("data/resampled_data.csv",header=None).values
    return data_resampled


def label_encoded(path):
    df = pd.read_csv(path,header=None)
    label = df.iloc[:, -1]
    label[label != "spammer"] = 0
    label[label == "spammer"] = 1
    data = df.iloc[:,:-1]
    new_data = pd.concat((data,label),axis=1)
    # new.to_csv("data/Twitter.csv",header=None,index=None)
    return new_data


def random_order(path):
    data = pd.read_csv(path,header=None,dtype=int)
    timestep = 40
    batchsize = data.shape[0] // timestep
    pos_data = data[:5000]
    neg_data = data[5000:]
    pos_batchsize = pos_data.shape[0] // timestep
    neg_batchsize = neg_data.shape[0] // timestep

    data = []
    for i in range(timestep):
        batchdata = pd.DataFrame(np.zeros([batchsize, pos_data.shape[1]]),dtype=int)
        pos_batch = pos_data[i * pos_batchsize:(i + 1) * pos_batchsize]
        pos_index = np.array(sample(range(batchsize), pos_batchsize))
        batchdata.iloc[pos_index] = pos_batch.values
        neg_index = list(set(range(batchsize)) - set(pos_index))
        neg_batch = neg_data[i * neg_batchsize:(i + 1) * neg_batchsize]
        batchdata.iloc[neg_index] = neg_batch.values
        data.append(batchdata)
    # data.to_csv("Twitter.csv", header=None, index=None)
    return data


def getClassColors():
    """
    Returns various different colors.
    """
    return np.array(['#000080','#00CC01','#FFCCCC','#ACE600',  '#2F2F2F', '#8900CC', '#0099CC',
                     '#00CC01','#915200', '#D9007E',  '#5E6600', '#FFFF00',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED', '#915200', '#999999',
                     '#0000FF', '#FF0000',  '#2F2F2F', '#8900CC', '#0099CC',
                     '#ACE600',  '#FFCCCC', '#5E6600', '#FFFF00', '#999999',
                     '#FF6000', '#00FF00', '#FF00FF', '#00FFFF', '#FFFF0F', '#F0CC01',
                     '#9BC6ED','#FFA500','#D9007E'])


def animation_with_label(out_prob,y,xx,yy,Z):


    # build plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)


    # build static plot for z
    x = out_prob
    # plot points
    cm_dark = mpl.colors.ListedColormap(getClassColors())
    sca = ax.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=2, cmap=cm_dark,alpha = 0.8)
    # ax.set_title('time step {0}'.format(0))
    ax.set_xlabel('Z1')
    ax.set_ylabel('Z2')
    ax.set_xlim(-4,10)
    ax.set_ylim(-10,10)

    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    # plot decision boundary
    # xx,yy,Z = decision_bound
    # Z = Z.reshape(xx.shape)
    # ax2.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.3)
    # build dynamic plot
    def update(i):
        # label = 'timestep {0}'.format(i)
        # print(label)
        x = out_prob
        data = [[x1,x2] for x1, x2 in zip(x[:,0],x[:,1])]
        sca.set_offsets(data)
        sca.set_array(y.ravel())
        # cm_dark = mpl.colors.ListedColormap(['r', 'g'])
        # plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=cm_dark)
        ax.set_title('time step {0}'.format(i))
        # label = 'timestep {0}'.format(i)
        # print(label)
    # anim = FuncAnimation(fig, update,frames= range(len(batchdata)),interval=200)

    # Set up formatting for the movie files
    # anim.save('Data/hyperplane/zx_plot.mp4', writer="ffmpeg")
    # plt.show()
    return fig

def plot_loss(train_loss,val_loss):
    plt.plot(range(len(train_loss)),train_loss,label = "validation loss")
    plt.plot(range(len(train_loss)),val_loss,label = "training loss")
    ax = plt.axes()
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss value")
    plt.title("loss iteration")
    plt.legend()
    plt.savefig("result/loss.jpg")
    plt.show()

def confusionMatrix(y_pred,y):
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    precision = tp * 1.0 / (tp + fp)
    recall = tp * 1.0 / (tp + fn)
    fmeasure = (2 * precision * recall) / (precision + recall +1e-3)
    result = [precision, recall, fmeasure]
    return result

if __name__ == '__main__':
    # data = label_encoded("data/twitter.csv")
    data = random_order("data/Twitter.csv")
