import numpy as np
import matplotlib.pyplot as plt

# with open("test.txt","r") as file:
#     while(1):
#         line = file.readline()
#         if(not line):
#             break
#         pre = line.split(",")
#         line = file.readline()
#         tar = line.split(",")
#         pre = np.array(pre[:-1]).reshape([5,512])
#         tar = np.array(tar[:-1]).reshape([5,512])
#         f, (a0) = plt.subplots(1, sharex=False, sharey=False)
#         for p in range(5):
#             pp = np.zeros([512,2])
#             pp[:,0] = pre[p] #blue
#             pp[:,1] = tar[p] #yellow
#
#             a0.plot(pp)
#
#             plt.show()

P = np.zeros([6,102400])
T = np.zeros([6,102400])
with open("pre.txt","r") as file:
    data = file.readlines()
    for index,d in enumerate(data):
        temp = d.split(",")
        P[index,:] = np.array(temp[:-1]).astype(np.int32)
with open("tar.txt","r") as file:
    data = file.readlines()
    for index,d in enumerate(data):
        temp = d.split(",")
        T[index,:] = np.array(temp[:-1]).astype(np.int32)
for i in range(6):
    f, (a0) = plt.subplots(1, sharex=False, sharey=False)
    a0.plot(P[i],color="orange")
    a0.plot(T[i],color="blue")
    plt.show()
