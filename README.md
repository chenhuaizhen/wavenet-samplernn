# wavenet-samplernn
关于wavenet和samplernn的一些实验  
some experiments according to the wavenet and samplernn  

## Cite(引用)
wavenet:https://github.com/ibab/tensorflow-wavenet  
        [WaveNet generative neural network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)  
samplernn:https://github.com/Unisound/SampleRNN   
        [SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)  

## piano generate(钢琴乐生成)

## text generate(文本生成)
基于同样的原理，也可将文本信息用0-1数组表示进行训练，本次实验采用了《寻秦记》作为训练样本，除去特殊符号其共有4119种字，可用一个大小为4119的矩阵来表示其中的任意字。由于4119比之前的256大的多，所以可以先采用“字嵌入”的方式进行大小的缩减，其效果如下图（黄线为加入字嵌入，蓝线为4119大小输入）  
based on the same principle, wavenet or samplernn can be used to generate the text using the one-hot embedding  
in this experiment, I use a Chinese novel named "Xun Qin Ji" as training data, it contains 4119 kinds of Chinese characters except some special notations, so the size of final input matrix is [4119], much larger than 256 in previous experiments  
and I tried to use word embedding to decrease the size of input, the result like below(the yellow line means word-embedding, the other means the orignal input(4119))  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/1.jpg)
