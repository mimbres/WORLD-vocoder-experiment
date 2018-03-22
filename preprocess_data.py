#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data.py

Created on Fri Aug 18 03:48:32 2017

@author: sungkyun
"""

#%% 
# import
import pyworld as pw
import numpy as np
import librosa
from timeit import default_timer as timer
import os

# path
#data_dir = os.path.dirname(os.path.realpath(__file__)) + '/Data/'
data_dir = './Data/'


#%%
#x, fs = sf.read('/media/sungkyun/DATA Linux/DSD100/Sources/Test/040 - The Long Wait - Back Home To Blue/vocals.wav')
#x, fs = sf.read(data_dir + '120-bpm-E-Always-Kyte-Acapella-Copyright-freevocals.wav')
#x, fs = librosa.load(data_dir + '120-bpm-E-Always-Kyte-Acapella-Copyright-freevocals.wav', sr=33075, mono=True, dtype=np.float64)
x, fs = librosa.load(data_dir + 'Eyes Nose Lips Acapella Cover_short.wav', sr=33075, mono=True, dtype=np.float64)
#x, fs = librosa.load(data_dir + 'Eyes Nose Lips Acapella Cover_part3_chorus.wav', sr=33075, mono=True, dtype=np.float64)
#x, fs = librosa.load(data_dir + 'umbrella.wav', sr=33075, mono=True, dtype=np.float64)

if len(x.shape) > 1 : x = np.mean(x, axis=1)  # stereo to mono

#%% wav --> parameters
start = timer()
#_f0, t = pw.dio(x, fs)    # raw pitch extractor
#f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
f0, t = pw.harvest(x,fs)
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
end = timer()
print('Feature Extraction:', end - start, 'seconds')

# f0_new
from copy import deepcopy  # to avoid call by reference!! 
f0_new = deepcopy(f0)   # 1-58 59-138 139-198 // 269-360 // 429-522
f0_new[1:198] = np.flip(f0_new[1:198],0) # reverse pitch
f0_new[269:360] = f0_new[269:360] + 62 #E(330hz) -> G (392hz)
f0_new[429:522] = f0_new[429:522] + 193#E(330hz) -> G(523hz)

#%% reduce dimension of spectral envelope and aperiodicity.
enc_sp = pw.code_spectral_envelope(sp, fs, number_of_dimensions=32)
dec_sp = pw.decode_spectral_envelope(enc_sp, fs, fft_size=(sp.shape[1] - 1) * 2)

enc_ap = pw.code_aperiodicity(ap,fs)
dec_ap = pw.decode_aperiodicity(enc_ap, fs, fft_size=(ap.shape[1] - 1)*2)

#%%
y = pw.synthesize(f0, sp, ap, fs)
librosa.output.write_wav('y_EyesNose_short_resynthesis.wav', y, fs)
#%%
y = pw.synthesize(f0, dec_sp, ap, fs)
librosa.output.write_wav('y_EyesNose_short_resynthesis_sp_decode_32.wav', y, fs)

#%% synthesis using new f0
y = pw.synthesize(f0_new, sp, ap, fs)
librosa.output.write_wav('y_EyesNose_short_new_F0_sp_decode_32.wav', y, fs) 
