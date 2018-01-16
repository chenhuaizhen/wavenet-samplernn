import numpy as np
import io

def initDict(fileAdd):
    dict = {}
    reDict = {}
    with io.open(fileAdd, mode="r", encoding="utf-8") as file:
    # with open(fileAdd, "r") as file:
        txt = file.read()
        index = 0
        for t in txt:
            if (t not in dict):
                dict[t] = index
                reDict[index] = t
                index += 1
        return dict,reDict

def tranToData(data,dict):
    res = ""
    for d in data:
        res += dict[d]
    return res

def initStartCtx(fileAdd,dict):
    output = []
    with io.open(fileAdd, mode="r", encoding="utf-8") as file:
        txt = file.read()
        for t in txt:
            output.append(dict[t])
    output = np.array(output).astype(np.int32)
    return output

def initData(fileAdd):
    dict = {}
    reDict = {}
    with io.open(fileAdd, mode="r", encoding="utf-8") as file:
        txt = file.read()
        index = 0
        for t in txt:
            if (t not in dict):
                dict[t] = index
                reDict[index] = t
                index += 1
        print(len(dict),len(reDict)) # 4118
        return txt,dict

def tranToDict(data,dict):
    res = []
    for d in data:
        res.append(dict[d])
    return np.array(res)

def getData(data,length):
    index = random.randint(0, len(data)-1)
    start = random.randint(0,len(data[index])-length-2)
    return data[index][start:start+length]

def getBatchData(data,batch_size,length):
    batch_data = []
    for i in range(batch_size):
        batch_data.append(getData(data,length))
    batch_data = np.array(batch_data).reshape([batch_size,length,1])
    return batch_data

def initDict2(fileAdd):
    dict = {}
    reDict = {}
    with open(fileAdd, "r",encoding="UTF-8") as file:
        txt = file.read()
        index = 0
        for t in txt:
            if (t not in dict):
                dict[t] = index
                reDict[index] = t
                index += 1
        return dict,reDict

def initDict(dictAdd,reDictAdd):
    dict = {}
    reDict = {}
    d1 = np.load(dictAdd)
    d2 = np.load(reDictAdd)
    for i in d1.item():
        dict[i] = d1.item().get(i)
    for i in d2.item():
        reDict[i] = d2.item().get(i)
    return dict,reDict

saveDict = "dict.npy"
saveReDict = "reDict.npy"

dict,reDict = initDict2("data.txt")
dict2,reDict2 = initDict(saveDict,saveReDict)

# d = np.load("dict.npy")
# dd = {}
# for i in d.item():
#     dd[i] = d.item().get(i)
# print(len(dd))

with open("start.txt","r",encoding="UTF-8") as file:
    f = file.read()
    for ff in f:
        print(dict[ff],dict2[ff])
        print(reDict[dict[ff]],reDict2[dict2[ff]])
