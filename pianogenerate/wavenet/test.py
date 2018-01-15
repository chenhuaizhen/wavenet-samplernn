import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
(rate,sig) = wav.read("output.wav")
print(sig)