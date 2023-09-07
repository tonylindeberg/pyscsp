"""Extensions of parts of the discscsp package to PyTorch networks"""

import numpy as np
import math
from math import pi
from typing import Union
import torch

from pyscsp.discscsp import gaussfiltsize

# ==>> Import from other Python package awaiting a full PyTorch interface for
# ==>> the modified Bessel functions that determine the filter coefficients
# ==>> for the discrete analogue of the Gaussian kernel
from pyscsp.discscsp import make1Ddiscgaussfilter


"""Discrete Scale Space and Scale-Space Derivative Toolbox for PyTorch

For computing discrete scale-space smoothing by convolution with the discrete
analogue of the Gaussian kernel and for computing discrete derivative approximations
by applying central difference operators to the smoothed data. 

This code is the result of porting a subset of the routines in the Matlab packages
discscsp and discscspders to Python.

Note: The scale normalization does not explicitly compensate for the additional 
variance 1/12 for the integrated Gaussian kernel or the additional variance 1/6
for the linearly integrated Gaussian kernel.

References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
A basis for low-level feature detection", Journal of Mathematical Imaging and Vision, 
3(4): 349-376.

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.

Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.
"""


def make1Dgaussfilter(
        sigma : Union[float, torch.Tensor], # scalar PyTorch tensor if sigma is to be learned
        scspmethod : str = 'discgauss',
        epsilon : float = 0.01, #
        D : int = 1) -> torch.Tensor:
    """Generates a mask for discrete approximation of the Gaussian kernel 
by separable filtering, using either of the methods:

  'discgauss' - the discrete analogue of the Gaussian kernel
  'samplgauss' - the sampled Gaussian kernel
  'normsamplgauss' - the sampled Gaussian kernel
  'intgauss' - the integrated Gaussian kernel
  'linintgauss' - the linearily integrated Gaussian kernel

The discrete analogue of the Gaussian kernel has the best theoretical properties 
of these kernels, in the sense that it obeys both (i) non-enhancement of local 
extrema over a 2-D spatial domain and (ii) non-creation of local extrema from 
any finer to any coarser level of scale for any 1-D signal. The filter coefficents 
are (iii) guaranteed to be in the interval [0, 1] and do (iv) exactly sum to one 
for an infinitely sized filter. (v) The spatial standard deviation of the discrete 
kernel is also equal to the sigma value. The current implementation of the this filter 
in terms of modified Bessel functions of integer order is, however, not 
supported in terms of existing PyTorch functions, implying that the choice 
of this method will not allow for scale adaptation by backprop.

For this reason, the alternative methods 'samplgauss', 'normsamplgauss, 'intgauss' 
and 'linintgauss' are provided, with full implementations in terms of PyTorch
functions and thereby supporting scale adaptation by backprop.

For these methods, there are the possible advantages (+) and disadvantages (-):

  'samplgauss' + no added scale offset in the spatial discretization
               - the kernel values may become greater than 1 for small values of sigma
               - the kernel values do not sum up to one
               - for very small values of sigma the kernels have too narrow shape

  'normsamplgauss' + no added scale offset in the spatial discretization
                   + formally the kernel values are guaranteed to be in the interval [0, 1]
                   + formally the kernel values are guaranteed to sum up to 1 
                   - the complementary normalization of the kernel is ad hoc
                   - for very small values of sigma the kernels have too narrow shape

  'intgauss' + the kernel values are guaranteed to be in the interval [0, 1]
             + the kernel values are guaranteed to sum up to 1 over an infinite domain
             - the box integration introduces a scale offset of 1/12

  'linintgauss' + the kernel values are guaranteed to be in the interval [0, 1]
                - the triangular window integration introduces a scale offset of 1/6

The parameter epsilon specifies an upper bound on the relative truncation error
for separable filtering over a D-dimensional domain.
          
References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
"""
    if (scspmethod == 'samplgauss'):
        return make1Dsamplgaussfilter(sigma, epsilon, D)
    if (scspmethod == 'normsamplgauss'):
        return make1Dnormsamplgaussfilter(sigma, epsilon, D)
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
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    return gauss(x, sigma)


def gauss(x, sigma=1.0):
    return 1/(math.sqrt(2*pi)*sigma)*torch.exp(-(x**2/(2*sigma**2)))


def make1Dnormsamplgaussfilter(sigma, epsilon=0.01, D=1):
    prelfilter = make1Dsamplgaussfilter(sigma, epsilon, D)
    return prelfilter/torch.sum(prelfilter)


def make1Dintgaussfilter(sigma, epsilon=0.01, D=1):
    # Box integrated Gaussian kernel over each pixel support region
    # Remark: Adds additional spatial variance 1/12 to the kernel
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(z, sigma=1.0):
    return 1/2*(1 + torch.erf(z/(math.sqrt(2)*sigma)))


def make1Dlinintgaussfilter(sigma, epsilon=0.01, D=1):
    # Linearly integrated Gaussian kernel over each extended pixel support region 
    # Remark: Adds additional spatial variance 1/6 to the kernel
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)
    # The following equation is the result of a closed form integration of the expression
    # for the filter coefficients in Eq (2.89) on page 52 in Lindeberg's PhD thesis from 1991
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - 2*gauss(x, sigma) + gauss(x - 1, sigma))


def x_scaled_erf(x, sigma=1.0):
    return x * scaled_erf(x, sigma)


def jet2mask(C0=0.0, Cx=0.0, Cy=0.0, Cxx=0.0, Cxy=0.0, Cyy=0.0, sigma=1.0):
    """Returns a discrete mask for a Gaussian derivative layer according to

Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.

using variance-based normalization of the Gaussian derivative operators 
for scale normalization parameter gamma = 1
"""
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
