"""Scale Space Toolbox for PyTorch

Extend subsets of the discscsp and affscsp modules to PyTorch:

(discscp) For computing discrete scale-space smoothing by convolution with the discrete
analogue of the Gaussian kernel and for computing discrete derivative approximations
by applying central difference operators to the smoothed data. 

(affscsp) Functions for performing the equivalent effect of convolving an image with
discrete approximations of directional derivatives of affine Gaussian
kernels, including mechanisms for scale normalization as well as a mechanism
for relative normalization of receptive field responses between different orders 
of spatial diffentiation.

References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
A basis for low-level feature detection", Journal of Mathematical Imaging and Vision, 
3(4): 349-376.

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.

Lindeberg and Garding (1997) "Shape-adapted smoothing in estimation 
of 3-D depth cues from affine distortions of local 2-D structure",
Image and Vision Computing 15: 415-434

Lindeberg (2013) "A computational theory of visual receptive fields", 
Biological Cybernetics, 107(6): 589-635. (See Equation (69).)

Lindeberg (2021) "Normative theory of visual receptive fields", 
Heliyon 7(1): e05897: 1-20.

Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.
"""

import math
from math import pi
from typing import Union
import numpy as np
import torch

from pyscsp.discscsp import gaussfiltsize, variance1D

# ==>> Import from other Python package awaiting a full PyTorch interface for
# ==>> the modified Bessel functions that determine the filter coefficients
# ==>> for the discrete analogue of the Gaussian kernel
from pyscsp.discscsp import make1Ddiscgaussfilter

from pyscsp.affscsp import samplaffgausskernel, scnormaffdirdermask


def make1Dgaussfilter(
        # sigma should be a 0-D PyTorch tensor if sigma is to be learned
        sigma : Union[float, torch.Tensor],
        scspmethod : str = 'discgauss',
        epsilon : float = 0.01,
        D : int = 1
    ) -> torch.Tensor :
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
    kernel is also equal to the sigma value. The current implementation of the this 
    filter in terms of modified Bessel functions of integer order is, however, not 
    supported in terms of existing PyTorch functions, implying that the choice 
    of this method will not allow for scale adaptation by back propagation.

    For this reason, the alternative methods 'samplgauss', 'normsamplgauss, 'intgauss' 
    and 'linintgauss' are provided, with full implementations in terms of PyTorch
    functions and thereby supporting scale adaptation by back propagation.

    For these methods, there are the possible advantages (+) and disadvantages (-):

    'samplgauss' + no added scale offset in the spatial discretization
                 - the kernel values may become greater than 1 for small values of sigma
                 - the kernel values do not sum up to one
                 - for very small values of sigma the kernels have too narrow shape

    'normsamplgauss' + no added scale offset in the spatial discretization
                     + formally the kernel values are guaranteed to be in the 
                       interval [0, 1]
                     + formally the kernel values are guaranteed to sum up to 1 
                     - the complementary normalization of the kernel is ad hoc
                     - for very small values of sigma the kernels have too narrow shape

    'intgauss' + the kernel values are guaranteed to be in the interval [0, 1]
               + the kernel values are guaranteed to sum up to 1 over an infinite domain
               - the box integration introduces a scale offset of 1/12 at coarser scales

    'linintgauss' + the kernel values are guaranteed to be in the interval [0, 1]
                  - the triangular window integration introduces a scale offset 
                    of 1/6 at coarser scales

    The parameter epsilon specifies an upper bound on the relative truncation error
    for separable filtering over a D-dimensional domain.
          
    References:

    Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
    Pattern Analysis and Machine Intelligence, 12(3): 234--254.

    Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.
    """
    if scspmethod == 'discgauss':
        # ==>> Note! Here sigma is not PyTorch variable to allow for scale
        # ==>> adaptation by backprop. That would need a PyTorch interface
        # ==>> for the modified Bessel functions
        return torch.from_numpy(\
                (make1Ddiscgaussfilter(sigma, epsilon, D))).type(torch.FloatTensor)

    if scspmethod == 'samplgauss':
        return make1Dsamplgaussfilter(sigma, epsilon, D)

    if scspmethod == 'normsamplgauss':
        return make1Dnormsamplgaussfilter(sigma, epsilon, D)

    if scspmethod == 'intgauss':
        return make1Dintgaussfilter(sigma, epsilon, D)

    if scspmethod == 'linintgauss':
        return make1Dlinintgaussfilter(sigma, epsilon, D)

    raise ValueError(f'Scale space method {scspmethod} not implemented')


def make1Dsamplgaussfilter(
        sigma : Union[float, torch.Tensor],
        epsilon : float = 0.01,
        D : int = 1
    ) -> torch.Tensor :
    """Computes a 1D filter for separable discrete filtering with the 
    sampled Gaussian kernel.

    Note: At very fine scales, the variance of the discrete filter may be much 
    lower than sigma^2.
    """
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)

    return gauss(x, sigma)


def gauss(
        x : torch.Tensor,
        sigma : float = 1.0
    ) -> torch.Tensor :
    """Computes the 1-D Gaussian of a PyTorch tensor representing 1-D x-coordinates.
    """
    return 1/(math.sqrt(2*pi)*sigma)*torch.exp(-(x**2/(2*sigma**2)))


def make1Dnormsamplgaussfilter(
        sigma : torch.Tensor,
        epsilon : float = 0.01,
        D : int = 1
    ) -> torch.Tensor :
    """Computes a 1D filter for separable discrete filtering with the L1-normalized 
    sampled Gaussian kernel.

    Note: At very fine scales, the variance of the discrete filter may be much lower
    than sigma^2.
    """
    prelfilter = make1Dsamplgaussfilter(sigma, epsilon, D)

    return prelfilter/torch.sum(prelfilter)


def make1Dintgaussfilter(
        sigma : torch.Tensor,
        epsilon : float = 0.01,
        D : int = 1
    ) -> torch.Tensor :
    """Computes a 1D filter for separable discrete filtering with the box integrated 
    Gaussian kernel over each pixel support region, according to Equation (3.89) on 
    page 97 in Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.

    Note: Adds additional spatial variance 1/12 to the kernel at coarser scales.
    """
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)

    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(
        z : torch.Tensor,
        sigma : float = 1.0
    ) -> torch.Tensor :
    """Computes the scaled error function (as depending on a scale parameter sigma)
    of a PyTorch tensor representing 1-D x-coordinates.
    """
    return 1/2*(1 + torch.erf(z/(math.sqrt(2)*sigma)))


def make1Dlinintgaussfilter(
        sigma : torch.Tensor,
        epsilon : float = 0.01,
        D : int = 1
    ) -> torch.Tensor :
    """Computes a 1D filter for separable discrete filtering with the linearly 
    integrated Gaussian kernel over each extended pixel support region.

    Note: Adds additional spatial variance 1/6 to the kernel at coarser scales.
    """
    vecsize = int((math.ceil(1.0*gaussfiltsize(sigma, epsilon, D))))
    x = torch.linspace(-vecsize, vecsize, 2*vecsize+1)

    # The following equation is the result of a closed form integration of
    # the expression for the filter coefficients in Eq (3.90) on page 97
    # in Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + \
           x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - \
           2*gauss(x, sigma) + gauss(x - 1, sigma))


def x_scaled_erf(
        x : torch.Tensor,
        sigma : float = 1.0
    ) -> torch.Tensor :
    """Computes the product of the x-coordinate and scaled error function (as depending 
    on a scale parameter sigma) of a PyTorch tensor representing 1-D x-coordinates.
    """
    return x * scaled_erf(x, sigma)


def jet2mask(C0=0.0, Cx=0.0, Cy=0.0, Cxx=0.0, Cxy=0.0, Cyy=0.0, sigma=1.0):
    """Returns a discrete mask for a Gaussian derivative layer according to
    Equation (11) in

    Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
    networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.

    using variance-based normalization of the Gaussian derivative operators 
    for scale normalization parameter gamma = 1.

    Note: This function is a mere template for how to compute the Gaussian derivative
    layer. For efficiency reasons, it may be better to generate the masks as PyTorch
    tensors only once and for all in the Gaussian derivative layer, and then combining 
    those at each new call of a Gaussian derivative layer.
    """
    return C0 + sigma*(Cx*dxmask() + Cy*dymask()) + \
           sigma**2/2*(Cxx*dxxmask() + Cxy*dxymask() + Cyy*dyymask())


def dxmask():
    """Returns a mask for discrete approximation of the first-order derivative 
    in the x-direction.
    """
    return torch.from_numpy(np.array([[ 0.0, 0.0,  0.0], \
                                      [-0.5, 0.0, +0.5], \
                                      [ 0.0, 0.0,  0.0]])).type(torch.FloatTensor)


def dymask():
    """Returns a mask for discrete approximation of the first-order derivative 
    in the y-direction.
    """
    return torch.from_numpy(np.array([[0.0, +0.5, 0.0], \
                                      [0.0,  0.0, 0.0], \
                                      [0.0, -0.5, 0.0]])).type(torch.FloatTensor)


def dxxmask():
    """Returns a mask for discrete approximation of the second-order derivative 
    in the x-direction.
    """
    return torch.from_numpy(np.array([[0.0,  0.0, 0.0], \
                                      [1.0, -2.0, 1.0], \
                                      [0.0,  0.0, 0.0]])).type(torch.FloatTensor)


def dxymask():
    """Returns a mask for discrete approximation of the mixed second-order 
    derivative in the x- and y-directions.
    """

    return torch.from_numpy(np.array([[-0.25, 0.00, +0.25], \
                                      [ 0.00, 0.00,  0.00], \
                                      [+0.25, 0.00, -0.25]])).type(torch.FloatTensor)


def dyymask():
    """Returns a mask for discrete approximation of the second-order derivative 
    in the y-direction.
    """
    return torch.from_numpy(np.array([[0.0, +1.0, 0.0], \
                                      [0.0, -2.0, 0.0], \
                                      [0.0, +1.0, 0.0]])).type(torch.FloatTensor)


def filtersdev(pytorchfilter : torch.tensor) -> float :
    """Returns the actual spatial standard deviation of a 1-D PyTorch filter
    """
    return math.sqrt(variance1D(pytorchfilter.numpy()))


def makesamplaffgausskernel(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        N : int
    ) -> np.ndarray :
    """Computes a sampled affine Gaussian kernel of size N x N defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma*(-1) x/2)

    with the covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    parameterized as

    Cxx = sigma1^2 * cos(phi)^2 + sigma2^2 * sin(phi)^2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)^2 + sigma2^2 * cos(phi)^2

    References:

    Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.

    Lindeberg and Garding (1997) "Shape-adapted smoothing in estimation 
    of 3-D depth cues from affine distortions of local 2-D structure",
    Image and Vision Computing 15:415-434
    """
    return torch.from_numpy(samplaffgausskernel(sigma1, sigma2, phi, N)).type(torch.FloatTensor)


def makescnormaffdirdermask(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int
    ) -> np.ndarray :
    """Returns a discrete directional derivative approximation mask, such that
    application of this mask to an image that has been smoothed with a zero-order 
    affine Gaussian kernel gives an approximation of the scale-normalized 
    directional derivative response to the receptive field

    sigma1^phiorder sigma2^orthorder D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent the partial derivative operators in the 
    directions phi and its orthogonal direction orth, respectively, where it
    is assumed that convolution with g(x; Sigma), (with the covariance matrix 
    Sigma specified using the same values of sigma1, sigma2 and phi), is 
    computed outside of this function).

    The intention is that the mask returned by this function should be applied
    to affine Gaussian smoothed images. Specifically, for an image processing
    method that makes use of a filter bank of directional derivatives of 
    affine Gaussian kernels, the intention is that the computationally heavy
    affine Gaussian smoothing operation should be performed only once, and
    that different directional derivative approximation masks should then
    be applied to the same affine Gaussian smoothed image, thus saving
    a substantial amount of work, compared to applying full size affine
    Gaussian directional derivative masks for different choices of orders
    for the directional derivatives.

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (31)).
    """
    return torch.from_numpy( \
            scnormaffdirdermask(sigma1, sigma2, phi, phiorder, orthorder)).type(torch.FloatTensor)
