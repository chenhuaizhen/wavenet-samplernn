import matplotlib.pyplot as plt
import numpy as np

def getMean(arr,n):
    size = len(arr)//n
    output = np.zeros(size)
    for s in range(size):
        output[s] = np.mean(arr[s*n:(s+1)*n])
    return output

Len = 20000
def getData(add):
    with open(add, "r") as file:
        loss = []
        accuracy = []
        index = 0
        for line in file.readlines():
            if (index == Len):
                break
            if ("acc" not in line or "loss" not in line):
                continue
            l = line.split(",")
            loss.append(float(l[3]))
            accuracy.append(float(l[5][1:-2]))
            index += 1
        return loss,accuracy

Loss = np.zeros([Len//100,2])
Acc = np.zeros([Len//100,2])
l,a = getData("2-2.out")
Loss[:,0] = getMean(np.array(l),n=100)
Acc[:,0] = getMean(np.array(a),n=100)
l,a = getData("1-1.out")
Loss[:,1] = getMean(np.array(l),n=100)
Acc[:,1] = getMean(np.array(a),n=100)

f, (a0,a1) = plt.subplots(2, sharex=False, sharey=False)
a0.plot(Loss)
a1.plot(Acc)
plt.xlabel('iteration/100', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.show()