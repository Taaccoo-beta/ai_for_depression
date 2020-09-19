# --------------------------------------------------------
# get audio feature (MFCC)
# Licensed under The MIT License [see LICENSE for details]
# Written by 
# --------------------------------------------------------

#In this document,there are two kinds of method of extracting
#MFCC:1. By adopting librosa 2. By adopting Pytorch

import librosa
from librosa import display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
print(librosa.__version__) #0.6.0
print(np.__version__) #1.13.1
print(matplotlib.__version__)#3.1.0


