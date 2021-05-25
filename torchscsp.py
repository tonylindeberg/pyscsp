"""Extensions of parts of the discscsp package to Pytorch networks"""

import numpy as np
import math
from math import pi
import torch

from pyscsp.discscsp import gaussfiltsize

# ==>> Import from other Python package awaiting a full PyTorch interface for
# ==>> the modified Bessel functions that determine the filter coefficients
# ==>> for the discrete analogue of the Gaussian kernel
from pyscsp.discscsp import make1Ddiscgaussfilter

def make1Dgaussfilter(sigma, scspmethod='samplgauss', epsilon=0.01, D=1):
    if (scspmethod == 'samplgauss'):
        return make1Dsamplgaussfilter(sigma, epsilon, D)
    elif (scspmethod == 'discgauss'):
        # ==>> Note! Here sigma is not PyTorch variable to allow for scale adaptation by backprop
        # ==>> That would need a PyTorch interface for modified Bessel functions
        return torch.from_numpy(make1Ddiscgaussfilter(sigma, epsilon, D)).type(torch.FloatTensor)
    elif (scspmethod == 'intgauss'):
        return make1Dintgaussfilter(sigma, epsilon, D)
    elif (scspmethod == 'linintgauss'):
        return make1Dlinintgaussfilter(sigma, epsilon, D)
    else:
        raise ValueError('Scale space method %s not implemented' % scspmethod)
 
    
def make1Dsamplgaussfilter(sigma, epsilon=0.01, D=1):
    vecsize = int((np.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    return gauss(x, sigma)


def gauss(x, sigma=1.0):
    return 1/(math.sqrt(2*pi)*sigma)*torch.exp(-(x**2/(2*sigma**2)));


def make1Dintgaussfilter(sigma, epsilon=0.01, D=1):
    # Box integrated Gaussian kernel over each pixel support region
    # Remark: Adds additional spatial variance 1/12 to the kernel
    vecsize = int((np.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(z, sigma=1.0):
    return 1/2*(1 + torch.erf(z/(math.sqrt(2)*sigma)))


def make1Dlinintgaussfilter(sigma, epsilon=0.01, D=1):
    # Linearly integrated Gaussian kernel over each extended pixel support region 
    # Remark: Adds additional spatial variance 1/6 to the kernel
    vecsize = int((np.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    # The following equation is the result of a closed form integration of the expression
    # for the filter coefficients in Eq (2.89) on page 52 in Lindeberg's PhD thesis from 1991
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - 2*gauss(x, sigma) + gauss(x - 1, sigma))


def x_scaled_erf(x, sigma=1.0):
    return x * scaled_erf(x, sigma)


def jet2mask(C0=0.0, Cx=0.0, Cy=0.0, Cxx=0.0, Cxy=0.0, Cyy=0.0, sigma=1.0):
    # Only variance-based normalization so far
    return C0 + sigma*(Cx*dxmask() + Cy*dymask()) + sigma**2/2*(Cxx*dxxmask() + Cxy*dxymask() + Cyy*dyymask())


def dxmask():
    return torch.from_numpy(np.array([[ 0.0, 0.0,  0.0], \
                                      [-0.5, 0.0, +0.5], \
                                      [ 0.0, 0.0,  0.0]]))


def dymask():
    return torch.from_numpy(np.array([[0.0, +0.5, 0.0], \
                                      [0.0,  0.0, 0.0], \
                                      [0.0, -0.5, 0.0]]))


def dxxmask():
    return torch.from_numpy(np.array([[0.0,  0.0, 0.0], \
                                      [1.0, -2.0, 1.0], \
                                      [0.0,  0.0, 0.0]]))


def dxymask():
    return torch.from_numpy(np.array([[-0.25, 0.00, +0.25], \
                                      [ 0.00, 0.00,  0.00], \
                                      [+0.25, 0.00, -0.25]]))


def dyymask():
    return torch.from_numpy(np.array([[0.0, +1.0, 0.0], \
                                      [0.0, -2.0, 0.0], \
                                      [0.0, +1.0, 0.0]]))


if __name__ == '__main__': 
    main() 
