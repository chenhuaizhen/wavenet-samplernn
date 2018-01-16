import numpy as np
import tensorflow as tf
import io
from wavenet import WaveNetModel

batch_size = 1
len_of_data = 1024
dilations = [ 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
filter_width = 2
residual_channels = 32
dilation_channels = 32
skip_channels = 512
quantization_channels = 4118
# embedding_channels = 50
use_biases = True
modelAdd = "Model3/model.ckpt"
saveAdd = "output.txt"
saveDict = "dict.npy"
saveReDict = "reDict.npy"

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

def tranToData(data,dict):
    res = ""
    for d in data:
        res += dict[d]
    return res

def main():

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=batch_size,
        dilations=dilations,
        filter_width=filter_width,
        residual_channels=residual_channels,
        dilation_channels=dilation_channels,
        skip_channels=skip_channels,
        quantization_channels=quantization_channels,
        # embedding_channels=embedding_channels,
        use_biases=use_biases)

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba_incremental(samples)

    # sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)

    # variables_to_restore = {
    #     var.name[:-2]: var for var in tf.global_variables()
    # }
    # for v in variables_to_restore:
    #     print(v)
    # saver = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()
    saver.restore(sess, modelAdd)
    # sess.run(net.init_ops)

    decode = samples

    Dict, resDict = initDict(saveDict, saveReDict)

    waveform = [3.]

    for step in range(len_of_data):
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]
        sample = np.random.choice(
            np.arange(quantization_channels), p=prediction)
        waveform.append(sample)

        print(step,len_of_data,sample)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as a wav file.
    out = sess.run(decode, feed_dict={samples: waveform})
    output = np.array(out).astype(np.int32)
    result = tranToData(output,resDict)
    # result = np.ceil(32768*np.sign(output/128)*((np.power(256,np.abs(output/128))-1)/255)).astype(np.int16)
    print("start^^^^^^^")
    with io.open(saveAdd,mode="a",encoding="utf-8") as file:
        file.write(result)
    print('Finished generating.')


if __name__ == '__main__':
    main()
