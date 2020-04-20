from DataHelper import *
from KL_calculation import Representation
from Multiscale_Drift_Detection_Tests import DriftDetection
import time

# data = DataHelper("data/Twitter.csv")
# rp = Representation(data.getdata())
# KLdistance = rp.cal_KLdivergence()
KLdistance = pd.read_csv("result/KLdivergence_class1.csv").iloc[:,:-1]
parameters = {"windowsize":30,"batchsize":None}
combine = np.zeros(KLdistance.shape)
t0 = time.time()
for i in range(KLdistance.shape[1]):
    f_distance = KLdistance.iloc[:,i]
    MDDT = DriftDetection(parameter=parameters)
    print ("feature:",(i+1))
    t_start = 0
    for j in range(f_distance.shape[0]):
        result,t_star,t_claim = MDDT.drift_detection(f_distance[t_start:j])
        if result:
            combine[t_star:t_claim,i] = 1
            t_start = t_star
print (time.time() - t0)
final = pd.DataFrame(combine)
# final.to_csv("no_combine_result.csv",index=False,header=False)

