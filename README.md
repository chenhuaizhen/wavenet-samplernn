# wavenet-samplernn
关于wavenet和samplernn的一些实验  
some experiments according to the wavenet and samplernn  

## Cite(引用)
wavenet:  
code:https://github.com/ibab/tensorflow-wavenet  
blog:[WaveNet generative neural network architecture](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)  
samplernn:  
code:https://github.com/Unisound/SampleRNN   
paper:[SampleRNN: An Unconditional End-to-End Neural Audio Generation Model](https://arxiv.org/abs/1612.07837)  

## Requirements(环境)
python 2-7  
tensorflow 1.3.0

## Piano generate(钢琴乐生成)
### Datasets(数据来源)
网上资料，来源于[sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017)，点击[下载](https://drive.google.com/drive/folders/0B7riq_C8aslvbWJuMGhJRFBmSHM)  
the datasets come from [sampleRNN_ICLR2017](https://github.com/soroushmehr/sampleRNN_ICLR2017), click [here](https://drive.google.com/drive/folders/0B7riq_C8aslvbWJuMGhJRFBmSHM) to download  

### Data preprocessing(数据预处理)
```python
def _normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def initData(data_add):
    test_data = np.load(data_add)
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
#### Rationality(合理性)
下面两个波形图分别代表处理前后的音频，虽然有些微差别，但几乎一致  
the following two graph represent the audio data before or after preprocessing, although have slight differences, they are nearly the same  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/1.jpg)

#### Availability(有效性)
经过预处理的数据会更趋近统一，比如原先（……,12,13,14,15,……）与（……,19,20,21,22,……）不同，经过处理后会出现相同的片段（……,x,1,1,1,……）中的（1,1,1）  
after preprocessing the data will tend to be unity, such as the orignal data part (……,12,13,14,15,……) is different from (……,19,20,21,22,……), but preprocessing will generate a common part (1,1,1)  
下图分别是有预处理（蓝色）和没有预处理（黄色）的Loss曲线  
the image below seperately means the loss curve about training with preprocessing(the blue one) or without preprocessing(the orange one)  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/2.jpg)

### Training(训练过程)
运行以下代码即可训练  
just run the code like  
```python
python train.py
```

#### 3-tier-sampleRNN
```python
batch_size = 32 # the more the better
big_frame_size = 8
frame_size = 2
q_levels = 256
rnn_type = 'GRU'
rnn_dim = 512 # prefer 1024 if your memory is enough
n_rnn = 1
len_of_data = 520*200
len_of_bptt = 520 # should be an integer multiple of big_frame_size(8)
```

#### 2-tier-sampleRNN
```python
batch_size = 64 # the more the better
frame_size = 16
q_levels = 256
rnn_type = 'GRU'
rnn_dim = 1024
n_rnn = 3
len_of_data = 512*200
len_of_bptt = 512 # should be an integer multiple of frame_size(16)
```

#### wavenet
```python
batch_size = 2 # the more the better
rate_of_wav = 16000
len_of_data = rate_of_wav * 5
dilations = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
filter_width = 2
residual_channels = 32
dilation_channels = 32
skip_channels = 512
quantization_channels = 256
```

### Samples(生成样本)
运行以下代码即可生成  
just run the code like  
```python
python generate.py
```

#### 3-tier-sampleRNN
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample2.wav)  
[sample3](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample3.wav)  

#### 2-tier-sampleRNN
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample2.wav)  
[sample3](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample3.wav)  
[sample4](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample4.wav)  

#### wavenet
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/wavenet/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/wavenet/sample2.wav)  

## Text generate(文本生成)
### Datasets(数据来源)
[寻秦记](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/textgenerate/data.txt)  

### Data preprocessing(数据预处理)
已将所有标点符号统一成'.'，训练过程中会自动生成"dict.npy"和"reDict.npy"保存字符与[0-4117]数组的映射关系    
already unify all the punctuation into dot '.' and it will generate the 'dict.npy' and 'resDict.npy' automatically which save the one-hot word-embedding correlation during the training time  

### Training(训练过程)
运行以下代码即可训练  
just run the code like  
```python
python train.py
```

#### 3-tier-sampleRNN
```python
batch_size = 32 # the more the better
big_frame_size = 8
frame_size = 2
q_levels = 4118
rnn_type = 'GRU'
rnn_dim = 512 # prefer 1024 if your memory is enough
n_rnn = 1
len_of_data = 520
emb_size = 50
```

#### 2-tier-sampleRNN
```python
batch_size = 32 # the more the better
frame_size = 16
q_levels = 4118
rnn_type = 'GRU'
rnn_dim = 1024
n_rnn = 3
len_of_data = 1024
emb_size = 50
```

#### wavenet
```python
batch_size = 64 # the more the better
len_of_data = 1024
dilations = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
filter_width = 2
residual_channels = 32
dilation_channels = 32
skip_channels = 512
quantization_channels = 4118
embedding_channels = 256
```

### Word-embedding(字嵌入)
基于同样的原理，也可将文本信息用0-1数组表示进行训练，本次实验采用了《寻秦记》作为训练样本，除去特殊符号其共有4118种字，可用一个大小为4118的矩阵来表示其中的任意字。由于4118比之前的256大的多，所以可以先采用“字嵌入”的方式进行大小的缩减，对应代码在model.py  
based on the same principle, wavenet or samplernn can be used to generate the text using the one-hot embedding  
in this experiment, I use a Chinese novel named "Xun Qin Ji" as training data, it contains 4118 kinds of Chinese characters except some special notations, so the size of final input matrix is [4118], much larger than 256 in previous experiments  
and I tried to use word embedding to decrease the size of input, the code is in model.py  

#### sampleRNN
```python
sample_shap=[tf.shape(sample_input_sequences)[0],tf.shape(sample_input_sequences)[1]*self.emb_size,1]
embedding = tf.get_variable("embedding", [self.q_levels, self.emb_size])
sample_input_sequences = tf.one_hot(tf.reshape(sample_input_sequences, [-1]),depth=self.q_levels,dtype=tf.float32)
sample_input_sequences = tf.matmul(sample_input_sequences, embedding)
sample_input_sequences = tf.reshape(sample_input_sequences,sample_shap)
```

#### wavenet
```python
def _create_embedding_layer(self, input_batch, in_channels, out_channels):
    input_batch = tf.reshape(input_batch, [-1, self.quantization_channels])
    with tf.name_scope('word_embedding'):
        weights_filter = self.variables['word_embedding']['filter'] # [quantization_channels,embedding_channels]
        encoded = tf.matmul(input_batch, weights_filter)
        shape = [self.batch_size, -1, self.embedding_channels]
        encoded = tf.reshape(encoded, shape)
        return encoded
```
但是效果并不明显  
but the result was not better than that without using word-embedding  

### Samples(生成样本)
运行以下代码即可生成  
just run the code like  
```python
python generate.py
```

#### 3-tier-sampleRNN
the start text is "项少龙大叹倒霉."  
[sample](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/textgenerate/3-tier-samplernn/sample.txt)  
```python
项少龙大叹倒霉.希望自己的马车扑前.仍未跨过门.苦恼道.谁也很有关项少龙消息.吻了赵致奇道.董匡.你很快照人.不由暗笑.琴清狠狠瞪了滕翼正在等待会我们的心.我要助他亲热时.乌卓或陵见先生请教田单如此挡我生擒赵穆拍案叫绝.所以才一点点.脑海里必然谁敢自己的威风好吗.赵盘沉吟着道.不知道.你究竟有人同时软逐一喜胜.城阳夫人也在险地全力抵着后.拍案叫绝.国家人.纪嫣然回来见我项少龙苦笑道.鹿丹身穿的肖月潭显然.玉坠其境.同时既是如何破她的事.索性势.又泛着楼无心道.他起这点.本人明晚被秦国.赵雅不会加起大妹子休想有两人打消了好他有利的男人握进成来.与你商量清醒来.垂头道.你不是那么奇印的一声叫道.你心似抱步般锐后.说不定可谓出话来.赵雅等闻言欢人.你是多么气概不如.手法.所以他既有个问题.不过衣食指派几个人就知项少龙故意原好整以暇过去.她一件事.先是因我们一把长剑开承认道.你可包在心不自一人包括了浪是非常危险.以在此时他.虽及不上人的才女.平原林建立成对弄她的手势.摆明当不错了向赵穆和项少龙想起她还有甚么事了.她们到他若撤不尽.但赵雅卑响地看着他就在场上齐境单却追上我们将亦颇感触和拇指叫容.他就算了.赵穆摇头叹道.你当场鬼足了菜道
```

#### 2-tier-sampleRNN
the start text is "小张和蛮牛与他们同属第七特种团队"  
[sample](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/textgenerate/2-tier-samplernn/sample.txt)  
```python
小张和蛮牛与他们同属第七特种团队之势.尤其一天气子任表示了一个钮子.反对所长.由调活一步出路长阶缩上他看.打量乌廷芳呢声看看了一理还是不大多对付赵穆.因为急白了大秘福地位对娘的神态.欣然道.君主师他会兼并毛.必敢立好.说王贲.不过群俊到形立着.寒焰熹感.才被剑鞘早给那脸前站地道.只求荆善乘机.答道.还在这里风斗礼活.泄漏出来的冰扣.项少龙感觉上次要问谁了.要找了你.最重了我吗.我休想第三个马材.本的魅力.项少龙却是一个宝刀.然后受的纤手出无出了半.充满着纤手而起.高起河胡.兼之过桑门机攻.但天罗片规被中牟茂矩.担缓晕过队事的迎来.项少龙叫头雾中金就有点执剑.雅夫人扮成多人奉陪他去前的得世.把地向主偷出两个糊涂.舒儿莫言八妹会在齐人被贪权的非甚么呢.我曾想真想通非徒.这刻.但项少龙心胸低声道.乌氏惈改变门而行实起了房的都可以干甚么惊少.足踝道.雅夫人可凭感愕然.项少龙心中.欣然道.以其实神项郎激的簇拥下了出奉衣装沫.她眼光不表起来.站在他挂绵退.柔船.项少龙想着地图席厉害.却欲弱分两晋.公平暗数.除了主席训字.国以部在搂过口.让鄙视在田单显然了.那一眼.若说王恩宠和.我就是异日子败害了.连昔年才伤三哥手.道.田建更可安.勾魂言趣道.项惘比香鼓般道.昨夜不带了一字.目四宣布往蒙了一个时辰.项少龙跳起来道.凤菲淡然离去.项少龙骇然道.上将候.本是说出来时二军.赵国都未撤底.此人冷静地上他肩天惊.有些孽宫.与项少龙早记着身体里.所有有单儿.对我就叫榻立了.这时两名武士的贵应任刺昏君.吕不韦身前.脸带项少龙百战.不讲后喁死了几个押时.坐下马儿挣扎只要巴夜的环镇剑滑.人.俏脸登时鼓响.可要塞握了赵穆.登时心中升着款喜插后.项少龙向坐满室走.囔道.少龙道.上我的乃俨望名的人.赵人低声道.魏贵惜对与成为这件.连那真.他们今晚后.神秘忆雷动刀的柔声道.项郸会杀死了.项少龙无情欲魂.闪闪道.项少龙心中一软.所有制得诸人族.尤其无人不能会阻.脸埋进嫪家.立即收买自己.环盈万精神中般来送大了风尘.项少龙坐入他唇背一眼.项少龙知道一旦有一个人如释呢.项少龙道.管不缚得见侯爷着.都担心不争追着这地步入房前挑手.让妾刀不望目.自主行说吧.她现在人家欢心之材.其中明白.教你把握还能公告吹书道.少龙剧明.其实唱是别馆经痕.只落开怀楼.将人剑术红的般间交待我好.没有人为齐王的滋味.众人忙道.不若了韩国.可住男下.教你知道无眼.赵妮电头大嚼.夫人想不
```

#### wavenet
the start text is "项少龙大叹倒霉.只想匆匆了事."  
[sample](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/textgenerate/wavenet/sample.txt)    
```python
项少龙大叹倒霉.只想匆匆了事.摆明是时候最严令.今晚立即赶返咸阳.均是鹿公的人.又稍勾有舒适.项少龙等一边.与项少龙三人分宾主坐好后.漫不经意后.他会在事情中吕不韦的夫君.嫪毐的气焰图妹所挂下来.这类.鹿公和储君.没有人面见到嫪菲们.比操期人非攻手之称.长枪像一表人物.看到确是大家绝.现在单美美请给王龁陪同把刀把敌.迫的是田某.但整个人怨恨及不上尽好.旋又噗哧娇笑.伍孚正不耐烦.到后面找不似别院.喳喳喳在他旁.托出这诸位大厅宫.众人完全绕到岸旁.徐先对这时刚加速又闻田氏以下的陷阱.藏到山脚口中亦救掉成功之.上去寻归.悄悄在下颌.与嫪毐的时间项大人能否搂紧了.却觉自己来到三十五个贤力.亦添于秦始师利争之眼.还叫起来找邱日升哈哈一笑.幸好我们的管中邪才来了.但肯做错吗.伍孚愕作耳边.双方都没有能却掌声在外几朝的敌场.天得非常震笑.如此进行伤之富号.欺清秀逸了一套.纪嫣然目光放流.迎身而起.加上精神时.项少龙心脏颇为恃汗.痛苦求他和嬴盈看得为他不动时.才女的事.锵.眼角全高亦有定脸.整个人吓得无惧无可不辩.人家给荆俊来了.所以既来追寻府子.本想的是城墙明晚的黑龙.项少龙和荆善等带成一类的山榻和众人成等人见他的存在.陷东邻森严的会议.来到同中邪讶异至极其人物物.各人大怒而止.一去都不来撩出自己的苦恼.说到底.先王乃因进一职位各方面的剑匠.名素宁.加上他的秀儿当作了这么一种图因.纪嫣然大感兴奋.因想像现在般说起来告诉嫣然.但谁都奉上嫪毐.又来此虚语.竟连溪眸赳张苦笑.知是要摆布待责任要滴头的项大人对管中邪非是非常人对寿宴附近.我们眼熟未定.贵儿之乌廷芳一众一大将来.还要他就不知如何赞笑.你们已商量.这大计早问下吧.项少龙心中暗笑.陪他过来抓了他的目光.心中惋惜.怎忍出兴迎涕.昌平君牵羊浅静.已远着马车倒在管中邪身上的声势.一窍莫通.绝望向大概.滕翼并肩道.批马虎背熊腰.形势图成.近人无不凡来.第一个时辰.研究三家的形势.均以此人于合理当.只要落着向桓齮娇躯剧颤地道.二哥对你来相朝吗.这小子这行客就是琴太傅来好整个焦荡.伍孚要清秀夫人似乎此时变成事.忽以想到善平而示抵一种迷人的弱人.不知是甚么剑手吧.纪嫣然见他的两人亦认取晶王后处是男子.接往蹄声沸涌.春花惶窄丰自联想到却有廉颇两人把蒲鶮感觉地收拾.项少龙低头了点.待田氏姊妹和成心之有深厚他对得.单美美则举杯道.渭南武士行馆的一群将领点.血色如电全到墙门下也.琴清俏目异响.连各
```
