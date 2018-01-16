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
3-tier-sampleRNN:  
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample2.wav)  
[sample3](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/3-tier-samplernn/sample3.wav)  

2-tier-sampleRNN:  
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample2.wav)  
[sample3](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample3.wav)  
[sample4](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/2-tier-samplernn/sample4.wav)  

wavenet:  
[sample1](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/wavenet/sample1.wav)  
[sample2](https://github.com/chenhuaizhen/wavenet-samplernn/blob/master/pianogenerate/wavenet/sample2.wav)  

## text generate(文本生成)
基于同样的原理，也可将文本信息用0-1数组表示进行训练，本次实验采用了《寻秦记》作为训练样本，除去特殊符号其共有4118种字，可用一个大小为4118的矩阵来表示其中的任意字。由于4118比之前的256大的多，所以可以先采用“字嵌入”的方式进行大小的缩减，其效果如下图（黄线为加入字嵌入，蓝线为4118大小输入）  
based on the same principle, wavenet or samplernn can be used to generate the text using the one-hot embedding  
in this experiment, I use a Chinese novel named "Xun Qin Ji" as training data, it contains 4118 kinds of Chinese characters except some special notations, so the size of final input matrix is [4118], much larger than 256 in previous experiments  
and I tried to use word embedding to decrease the size of input, the result like below(the yellow line means word-embedding, the other means the orignal input(4118))  
![image](https://github.com/chenhuaizhen/wavenet-samplernn/raw/master/image/1.jpg)

### samples(生成样本)
3-tier-sampleRNN:  
the start text is "项少龙大叹倒霉."  
```python
项少龙大叹倒霉.希望自己的马车扑前.仍未跨过门.苦恼道.谁也很有关项少龙消息.吻了赵致奇道.董匡.你很快照人.不由暗笑.琴清狠狠瞪了滕翼正在等待会我们的心.我要助他亲热时.乌卓或陵见先生请教田单如此挡我生擒赵穆拍案叫绝.所以才一点点.脑海里必然谁敢自己的威风好吗.赵盘沉吟着道.不知道.你究竟有人同时软逐一喜胜.城阳夫人也在险地全力抵着后.拍案叫绝.国家人.纪嫣然回来见我项少龙苦笑道.鹿丹身穿的肖月潭显然.玉坠其境.同时既是如何破她的事.索性势.又泛着楼无心道.他起这点.本人明晚被秦国.赵雅不会加起大妹子休想有两人打消了好他有利的男人握进成来.与你商量清醒来.垂头道.你不是那么奇印的一声叫道.你心似抱步般锐后.说不定可谓出话来.赵雅等闻言欢人.你是多么气概不如.手法.所以他既有个问题.不过衣食指派几个人就知项少龙故意原好整以暇过去.她一件事.先是因我们一把长剑开承认道.你可包在心不自一人包括了浪是非常危险.以在此时他.虽及不上人的才女.平原林建立成对弄她的手势.摆明当不错了向赵穆和项少龙想起她还有甚么事了.她们到他若撤不尽.但赵雅卑响地看着他就在场上齐境单却追上我们将亦颇感触和拇指叫容.他就算了.赵穆摇头叹道.你当场鬼足了菜道
```
2-tier-sampleRNN:  
the start text is "小张和蛮牛与他们同属第七特种团队"  
```python
小张和蛮牛与他们同属第七特种团队之势.尤其一天气子任表示了一个钮子.反对所长.由调活一步出路长阶缩上他看.打量乌廷芳呢声看看了一理还是不大多对付赵穆.因为急白了大秘福地位对娘的神态.欣然道.君主师他会兼并毛.必敢立好.说王贲.不过群俊到形立着.寒焰熹感.才被剑鞘早给那脸前站地道.只求荆善乘机.答道.还在这里风斗礼活.泄漏出来的冰扣.项少龙感觉上次要问谁了.要找了你.最重了我吗.我休想第三个马材.本的魅力.项少龙却是一个宝刀.然后受的纤手出无出了半.充满着纤手而起.高起河胡.兼之过桑门机攻.但天罗片规被中牟茂矩.担缓晕过队事的迎来.项少龙叫头雾中金就有点执剑.雅夫人扮成多人奉陪他去前的得世.把地向主偷出两个糊涂.舒儿莫言八妹会在齐人被贪权的非甚么呢.我曾想真想通非徒.这刻.但项少龙心胸低声道.乌氏惈改变门而行实起了房的都可以干甚么惊少.足踝道.雅夫人可凭感愕然.项少龙心中.欣然道.以其实神项郎激的簇拥下了出奉衣装沫.她眼光不表起来.站在他挂绵退.柔船.项少龙想着地图席厉害.却欲弱分两晋.公平暗数.除了主席训字.国以部在搂过口.让鄙视在田单显然了.那一眼.若说王恩宠和.我就是异日子败害了.连昔年才伤三哥手.道.田建更可安.勾魂言趣道.项惘比香鼓般道.昨夜不带了一字.目四宣布往蒙了一个时辰.项少龙跳起来道.凤菲淡然离去.项少龙骇然道.上将候.本是说出来时二军.赵国都未撤底.此人冷静地上他肩天惊.有些孽宫.与项少龙早记着身体里.所有有单儿.对我就叫榻立了.这时两名武士的贵应任刺昏君.吕不韦身前.脸带项少龙百战.不讲后喁死了几个押时.坐下马儿挣扎只要巴夜的环镇剑滑.人.俏脸登时鼓响.可要塞握了赵穆.登时心中升着款喜插后.项少龙向坐满室走.囔道.少龙道.上我的乃俨望名的人.赵人低声道.魏贵惜对与成为这件.连那真.他们今晚后.神秘忆雷动刀的柔声道.项郸会杀死了.项少龙无情欲魂.闪闪道.项少龙心中一软.所有制得诸人族.尤其无人不能会阻.脸埋进嫪家.立即收买自己.环盈万精神中般来送大了风尘.项少龙坐入他唇背一眼.项少龙知道一旦有一个人如释呢.项少龙道.管不缚得见侯爷着.都担心不争追着这地步入房前挑手.让妾刀不望目.自主行说吧.她现在人家欢心之材.其中明白.教你把握还能公告吹书道.少龙剧明.其实唱是别馆经痕.只落开怀楼.将人剑术红的般间交待我好.没有人为齐王的滋味.众人忙道.不若了韩国.可住男下.教你知道无眼.赵妮电头大嚼.夫人想不
```
wavenet:  
the start text is "致"  
```python
致神色却抱起拇指.只见剑打缠的衣服样儿.那大堂冲来了一场往视了一记后.避过多少服一会才回幽甲.拉开始注目韩当乱尽娇俏的女儿.脸容狠耳横赏.他到九公等如此威风正中.现在现在小盘撒娇糊涂.正是少龙升了不同的人.始时寒暄处.向着她这最好的温暖性的脖子藏身稍朝.到了这些地图先头.接着埋头狂徒华台.取礼娇妻.往后方上.燃出了一口气.阳光正火光古怪的肌肉和闪动.一辆气子退下马背.然后叹道.师傅.我死心不从了.醒过去了.这几句话不敢十足.策马发现在.才情愿去了.只兵将要和自招布成这方面旅.声音教他融化了这些旧人.不时露出感受.关秀女几句去.又露出各人挑选的感觉.难道朱姬若无其事.少龙与他深进没有重任.经惯.理过去时.把孙小姐立即遣得欲来.就是非常钦佩.项少龙环目一看.低声道.我们输却并不给我宰了吗.小盘亦可让他们安抵格心的肉体.立即他倒注视她的肉欲.心想女君给她国兴哀乖.道.她是谁.只是吕重此忙将就是那里依附我一统.这天自嫪外的魏的事.不由神态发出.不禁心中定好.扎例少以我这.便可算都不例管骥崖加影.至此长动迅速渡过高手.问道.那知是否该放心.更是露了做甚么知情究竟绝是无一色.不禁异然口涎她.唯有讶异无尽的返的黑暗.忽然间.纪嫣然道.少龙啊.他怎太久管先休婚间呢.蒲鶮回来时.特恩意意四溢的活天声不适.只见林内暗叫传来.可惜他要命不能宣盛成为首.外返题秦.那禁卫顺手飞城.心中欢喜.穿过乐怨的神色.听他动日处的杀害.思索打机.项少龙向走罪了入来.认定田氏姊妹在内.要知此刻已弄见回身色了.还然力扯紧他道.吕不韦给过你两年就是笨蛋.恕明天没有邀请由楼帐的实权妙妙行动.最顾忌才有李牧.吕不韦和盘字不胜往.教人收首满之.使婢仆参与之险致难以中牟更适合备的敌人.竟能绕心冷狞地欣悦道.你还有此有这么容易理好吗.区春都是一介是肯定指环.现在不会发觉睡了下去吧.他不晓不上.只认要他感到一声尖柔.但生出放矢般.到了一批石头.笑了起来.总是有定要头痛迎他.攀上因代表演身的观察力人胆不动压.项少龙正大感头痛.沉声道.可是针在一条混得不知情决是甚么预估过形多人最珍于冰汗.可纵我看不出是从大担巨.也不得先一人的劲机.纪嫣然道.在这刻是吗.赵倩此时不在骗她见到有庞大的侦查.不过口来韩咒横生.但却不谈里来.人说了初遇犯馆去凑茫这就只为他现实引人准备缠船.项少龙低声道.我又知道事.小盘自把所惑施了起来.默许片削瘦项家村.伏身而出.吓了一跳.项少龙悄悄看了她香唇无礼
```


