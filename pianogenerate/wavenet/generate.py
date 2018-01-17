import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from model import WaveNetModel

batch_size = 1
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
quantization_channels = 2**8
use_biases = True
modelAdd = "Model/model.ckpt"
saveAdd = "output.wav"

def getSample(pre,n=5):
    pIndex = pre.argsort()[-n:]
    pp = pre[pIndex]
    pp = pp/sum(pp)
    return np.random.choice(pIndex, p=pp)

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
        use_biases=use_biases)

    samples = tf.placeholder(tf.int32)

    next_sample = net.predict_proba_incremental(samples)

    sess.run(net.init_ops)

    saver = tf.train.Saver()
    saver.restore(sess, modelAdd)

    decode = samples

    start = [128]
    waveform = [start]

    for step in range(len_of_data):
        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window})[0]
        # sample = np.random.choice(
        #     np.arange(quantization_channels), p=prediction)
        sample = getSample(prediction,n=5)
        waveform.append(sample)

        print(step,len_of_data,sample-128)

    # Introduce a newline to clear the carriage return from the progress.
    print()

    # Save the result as a wav file.
    out = np.array(waveform)
    output = out[1:].astype(np.float32)
    output = (output - 128) * 100
    result = output.astype(np.int16)
    # result = np.ceil(32768*np.sign(output/128)*((np.power(256,np.abs(output/128))-1)/255)).astype(np.int16)
    wav.write(saveAdd,rate_of_wav,result)

    print('Finished generating.')


if __name__ == '__main__':
    main()
