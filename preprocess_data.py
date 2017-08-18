#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_data.py

Created on Fri Aug 18 03:48:32 2017

@author: sungkyun
"""

#%%
import pyworld as pw
import numpy as np
import soundfile as sf
from timeit import default_timer as timer

#%%
x, fs = sf.read('/media/sungkyun/DATA Linux/DSD100/Sources/Test/040 - The Long Wait - Back Home To Blue/vocals.wav')
x = np.mean(x, axis=1)  # stereo to mono

#%%
start = timer()
_f0, t = pw.dio(x, fs)    # raw pitch extractor
f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
y = pw.synthesize(f0, sp, ap, fs)
end = timer()

print(end - start)


#%%
sf.write('test/y_with_f0_refinement.wav', y, fs)
