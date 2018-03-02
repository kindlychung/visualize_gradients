from scipy.misc import imsave
from skimage.transform import resize
import numpy as np
import os
import pickle
def plot(activation_method):
    grad_dict = pickle.load(open(activation_method + ".pickle", "rb"))
    for k, v in grad_dict.items():
        img = np.vstack(v)
        img = img / np.max(np.abs(img)) # make sure it's within [-1, 1]
        h, w = img.shape
        w *= 5
        imsave(
            activation_method+k+".png",
            resize(img, (h, w)),
        )

for m in ["sigmoid", "tanh", "relu"]:
    plot(m)

