from __future__ import absolute_import, print_function

import os

import numpy as np
from torch.autograd import Variable


def tensor_to_np_img(tensor_img):
    tmp = (tensor_img.data).cpu().numpy()
    tmp1 = tmp[0, :, :, :]
    np_img = tmp1.transpose([1, 2, 0])
    return np_img


def truncate_image(img, s):
    # s: truncate size
    if(s>0):
        if(len(img.shape)==3):
            # F or C x H x W
            return img[:, s:(-s), s:(-s)]
        elif(len(img.shape)==4):
            # F x C x H x W
            return img[:, :, s:(-s), s:(-s)]
    else:
        return img


def tensor2numpy(tensor_in):
    """Transfer pythrch tensor to numpy array"""
    nparray_out = (Variable(tensor_in).data).cpu().numpy()
    return nparray_out


def transpose_kernel(k):
    """k for A(k)^T"""
    return np.fliplr(np.flipud(k))


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

# project image data to [0,1]
def box_proj(input):
    output = input
    output[output>1] = 1
    output[output<0] = 0
    return output
