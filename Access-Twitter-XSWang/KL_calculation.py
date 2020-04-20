from DataHelper import *
from collections import Counter
from scipy.stats import kurtosis, skew, entropy


class Representation:
    def __init__(self,data):
        self.data = data
        self.class_num = 2

    def cal_mean_by_class(self):
        data = self.data
        pos_mean_overall = []
        neg_mean_overall = []
        for t, batchdata in enumerate(data):
            pos = batchdata[batchdata.iloc[:,-1] == 1]  # spammer
            pos_mean = np.mean(pos.iloc[:, :-1], axis=0)
            neg = batchdata[batchdata.iloc[:,-1] == 0] # spammer
            neg_mean = np.mean(neg.iloc[:, :-1], axis=0)
            pos_mean_overall.append(pos_mean)
            neg_mean_overall.append(neg_mean)
        pos_mean_overall = np.array(pos_mean_overall)
        neg_mean_overall = np.array(neg_mean_overall)
        df = pd.DataFrame(pos_mean_overall)
        df.to_csv("result/pos_mean_overall.csv",header=False,index=False)
        df = pd.DataFrame(neg_mean_overall)
        df.to_csv("result/neg_mean_overall.csv",header=False,index=False)

    def cal_cosdist_by_eigenvect(self):
        data = self.data
        pos_dist_overall = []
        neg_dist_overall = []
        for t, batchdata in enumerate(data[:-1]):
            pos_dist = []
            neg_dist = []
            pos_present = vector_generation(batchdata,label = 1)
            pos_next = vector_generation(data[t+1], label = 1)
            neg_present = vector_generation(batchdata, label = 0)
            neg_next = vector_generation(data[t+1],label = 0)
            for f in range(pos_present.shape[0]):
                pos_dist.append(compute_similarity(pos_present[f,:], pos_next[f,:])[0])
                neg_dist.append(compute_similarity(neg_present[f,:],neg_next[f,:])[0])
            pos_dist_overall.append(pos_dist)
            neg_dist_overall.append(neg_dist)
        df_pos = pd.DataFrame(np.array(pos_dist_overall))
        df_pos.to_csv("result/pos_cos_distance.csv",header=None,index=None)
        df_neg = pd.DataFrame(np.array(neg_dist_overall))
        df_neg.to_csv("result/neg_cos_distance.csv", header=None, index=None)
        return df_pos,df_neg

    def cal_KLdivergence(self):
        data = self.data
        no_class = self.class_num
        for label in range(no_class):
            KL_overall = []
            for t, batchdata in enumerate(data[:-1]):
                currentbatch = batchdata.loc[batchdata.iloc[:,-1] == label]
                nextbatch = data[0].loc[data[t+1].iloc[:,-1] == label]
                KL = []
                for f in range(batchdata.shape[1]-1):
                    P = pd.DataFrame(Counter(currentbatch.values[:,f]).items(),columns=['x','px'],)
                    Q = pd.DataFrame(Counter(nextbatch.values[:,f]).items(),columns=['x','qx'])
                    PQ = pd.merge(P,Q,on='x',how='outer').fillna(0)
                    P = (PQ['px'].values + 0.5)/(np.sum(PQ['px']) + PQ.shape[0]//2)
                    Q = (PQ['qx'].values + 0.5)/(np.sum(PQ['qx'] + PQ.shape[0]//2))
                    KL_div = entropy(P,Q)
                    KL.append(KL_div)
                KL_overall.append(KL)

            KL_overall = pd.DataFrame(KL_overall)
            KL_overall.to_csv("result/KLdivergence_class%d.csv"%label,index=None,header=None)
        return KL_overall

def vector_generation(data,label = 0):
    data = data[data[:,-1] == label][:,:-1]
    mean = np.mean(data, axis=0, keepdims=True)
    variance = np.var(data, axis=0, keepdims=True)
    kurt = kurtosis(data).reshape([1, -1])
    skewness = skew(data).reshape([1, -1])
    entr = entropy(data).reshape([1, -1])
    Z = np.concatenate([mean, variance, kurt, skewness, entr], axis=0).T
    Z[Z == -np.inf] = 0
    Z = np.nan_to_num(Z, 0)
    return Z


def compute_similarity(a, b):
    similarity = 1 - np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
    return similarity.reshape(-1)


def main():
    data = DataHelper("data/Twitter.csv")
    # data.MinMaxScale()
    rp = Representation(data.getdata())
    # rp.cal_mean_by_class()
    # rp.cal_cosdist_by_eigenvect()
    _ = rp.cal_KLdivergence()


if __name__ == '__main__':
    main()