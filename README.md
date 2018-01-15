# wavenet-samplernn
关于wavenet和samplernn的一些实验  
some experiments according to the wavenet and samplernn  

## Cite(引用)
wavenet:https://github.com/ibab/tensorflow-wavenet  
        [WaveNet generative neural network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)  
samplernn:https://github.com/Unisound/SampleRNN   
        [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)  

## piano generate(钢琴乐生成)
### datasets(数据来源)
网上资料，来源于[sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017)，点击[下载](https://drive.google.com/drive/folders/0B7riq_C8aslvbWJuMGhJRFBmSHM)  
the datasets come from [sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017), click [here](https://drive.google.com/drive/folders/0B7riq_C8aslvbWJuMGhJRFBmSHM) to download  

### data preprocessing(数据预处理)
```python
def _normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def initData():
    test_data = np.load(test_data_add)
    test_data = _normalize(test_data)
    eps = np.float64(1e-5)
    test_data *= (255. - eps)
    test_data += eps / 2
    testData = test_data[:,1:] - test_data[:,:-1] + 128
    testData = test_data.astype(np.int32)
    return testData
```
其中有一步操作（下文称为预处理）可以使训练加速  
there is one step(which is named for preprocessing in the following paragraphs) can speed up the training process  
```python
testData = test_data[:,1:] - test_data[:,:-1] + 128
```
#### rationality(合理性)
下面两个波形图分别代表处理前后的音频，虽然有些微差别，但几乎一致  
the following two graph represent the audio data before or after preprocessing, although have slight differences, they are nearly the same  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/3.jpg)

#### availability(有效性)
经过预处理的数据会更趋近统一，比如原先（……,12,13,14,15,……）与（……,19,20,21,22,……）不同，经过处理后会出现相同的片段（……,x,1,1,1,……）中的（1,1,1）  
after preprocessing the data will tend to be unity, such as the orignal data part (……,12,13,14,15,……) is different from (……,19,20,21,22,……), but preprocessing will generate a common part (1,1,1)  
下图分别是有预处理（蓝色）和没有预处理（黄色）的Loss曲线  
the image below seperately means the loss curve about training with preprocessing(the blue one) or without preprocessing(the orange one)  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/2.jpg)

### samples(生成样本)
3-tier-sampleRNN:[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample1.wav)  
                 [sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample2.wav)  
                 [sample3](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample3.wav)  

## text generate(文本生成)
基于同样的原理，也可将文本信息用0-1数组表示进行训练，本次实验采用了《寻秦记》作为训练样本，除去特殊符号其共有4119种字，可用一个大小为4119的矩阵来表示其中的任意字。由于4119比之前的256大的多，所以可以先采用“字嵌入”的方式进行大小的缩减，其效果如下图（黄线为加入字嵌入，蓝线为4119大小输入）  
based on the same principle, wavenet or samplernn can be used to generate the text using the one-hot embedding  
in this experiment, I use a Chinese novel named "Xun Qin Ji" as training data, it contains 4119 kinds of Chinese characters except some special notations, so the size of final input matrix is [4119], much larger than 256 in previous experiments  
and I tried to use word embedding to decrease the size of input, the result like below(the yellow line means word-embedding, the other means the orignal input(4119))  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/1.jpg)
