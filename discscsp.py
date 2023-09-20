"""Discrete Scale Space and Scale-Space Derivative Toolbox for Python

For computing discrete scale-space smoothing by convolution with the discrete
analogue of the Gaussian kernel and for computing discrete derivative approximations
by applying central difference operators to the smoothed data. Then, different
types of feature detectors can be defined by combining discrete analogues of the
Gaussian derivative operators into differential expressions.

This code is the result of porting a subset of the routines in the Matlab packages
discscsp and discscspders to Python.

Note: The scale normalization does not explicitly compensate for the additional 
variance 1/12 for the integrated Gaussian kernel or the additional variance 1/6
for the linearly integrated Gaussian kernel.

References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
A basis for low-level feature detection", Journal of Mathematical Imaging and 
Vision, 3(4): 349-376.

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.

Lindeberg (1998a) "Feature detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 77-116.

Lindeberg (1998b) "Edge detection and ridge detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 117-154.

Lindeberg (2009) "Scale-space". In: B. Wah (Ed.) Wiley Encyclopedia of Computer 
Science and Engineering, John Wiley & Sons, pp. 2495-2504.

Compared to the original Matlab code, the following implementation is reduced 
in the following ways:
- there is no handling of scale normalization powers gamma that are not equal to one
- Lp-normalization is only implemented for p = 1
- much fewer functions of the N-jet have so far been implemented
- there is no passing of additional parameters to functions of the N-jet
- this reimplementation has not yet been thoroughly tested
"""
import numpy as np
from scipy.ndimage import correlate1d, correlate
from scipy.special import ive
from math import sqrt, exp, ceil, pi
from scipy.special import erf, erfcinv
from typing import NamedTuple, Union, List


# Object for storing the parameters of a scale-space discretization method
class ScSpMethod(NamedTuple):
    # either 'discgauss', 'samplgauss', 'normsamplgauss', 'intgauss' or 'linintgauss'
    methodname: str 
    epsilon: float

    
# Object for storing the parameters of a discretization method for computing
# scale-normalized derivatives, including also the necessary parameters of the
# underlying method for discrete approximation of the Gaussian smoothing operation
class ScSpNormDerMethod(NamedTuple):
    scspmethod: ScSpMethod
    normdermethod: str # either 'nonormalization', 'varnorm' or 'Lpnorm'
    gamma: float


def scspnormdermethodobject(
        scspmethod : str = 'discgauss',
        normdermethod : str = 'Lpnorm',
        gamma : float = 1.0,
        epsilon : float = 0.00000001
) -> ScSpNormDerMethod :
    return ScSpNormDerMethod(ScSpMethod(scspmethod, epsilon), normdermethod, gamma)


def scspconv(
        inpic,
        sigma : float,
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Computes the scale-space representation of the 2-D image inpic (or a 1-D signal) 
at scale level sigma in units of the standard deviation of the Gaussian kernel that 
is approximated discretely with the method scspmethod and with the formally infinite 
convolution operation truncated at the tails with a relative approximation error 
less than epsilon.

The following discrete approximation methods have been implemented:
  'discgauss' - the discrete analogue of the Gaussian kernel
  'samplgauss' - the sampled Gaussian kernel
  'normsamplgauss' - the normalized sampled Gaussian kernel
  'intgauss' - the integrated Gaussian kernel
  'linintgauss' - the linearily interpolated and integrated Gaussian kernel

The discrete analogue of the Gaussian kernel has the best theoretical properties 
of these kernels, in the sense that it obeys both (i) non-enhancement of local 
extrema over a 2-D spatial domain and (ii) non-creation of local extrema from 
any finer to any coarser level of scale for any 1-D signal. The filter coefficents 
are (iii) guaranteed to be in the interval [0, 1] and do (iv) exactly sum to one 
for an infinitely sized filter. (v) The spatial standard deviation of the discrete 
kernel is also equal to the sigma value.

The different methods have the possible advantages (+) and disadvantages (-):

  'discgauss' + guarantees non-enhancement of local extrema over a 2-D image domain
              + guarantees non-creation of new extrema from any finer to any
                coarser level of scale over a 1-D signal domain
              + the kernel values are guaranteed to be in the interval [0, 1]
              + the kernel values are guaranteed to sum to 1 over an infinite domain
              + no scale offset at all in the spatial discretization

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
                - the triangular window integration introduces a scale offset of 1/6
                  at coarser scales

Besides being a string, the argument scspmethod may also be an object
having the attributes scspmethod.methodname and scspmethod.epsilon.

The parameter epsilon specifies an upper bound on the relative truncation error
for separable filtering over a D-dimensional domain.
"""
    if (isinstance(scspmethod, str)):
        scspmethodname = scspmethod
    else:
        scspmethodname = scspmethod.methodname
        epsilon = scspmethod.epsilon

    if (scspmethodname == 'discgauss'):
        outpic = discgaussconv(inpic, sigma, epsilon)
    elif (scspmethodname == 'samplgauss'):
        outpic = samplgaussconv(inpic, sigma, epsilon)
    elif (scspmethodname == 'normsamplgauss'):
        outpic = normsamplgaussconv(inpic, sigma, epsilon)
    elif (scspmethodname == 'intgauss'):
        outpic = intgaussconv(inpic, sigma, epsilon)        
    elif (scspmethodname == 'linintgauss'):
        outpic = linintgaussconv(inpic, sigma, epsilon)        
    else:
        raise ValueError('Scale space method %s not implemented' % scspmethodname)

    return outpic


def scspconv_mult(
        inpic,
        sigmavec : List[float],
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Perform a similar function as the function scscpconv, with the 
difference that an array of sigma values can be provided instead of a 
single value, and that the cascade smoothing property of Gaussian 
convolution is then made use of to perform the computational more
efficiently than if applying the scspconv function for each
scale level separately.

eNote, however, that the cascade smoothing property works best when
using the discrete analogue of the Gaussian kernel 'discgauss', for
which the cascade smoothing property holds exactly in the ideal
case of an infinite image domain with kernels the smoothing kernels
having infinite support. For the otherw ways of approximating the
Gaussian kernel discretely, there may be deviations depending on
the scale values.

The output will be an array of one dimension more than for the function scspconv.
"""
    ndim = inpic.ndim

    if (ndim == 1):
        outpic = np.zeros((len(sigmavec), len(inpic)))
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :] = smoothpic[:]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :] = smoothpic[:]

    elif (ndim == 2):
        outpic = np.zeros((len(sigmavec),) + inpic.shape)
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :, :] = smoothpic[:, :]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :, :] = smoothpic[:, :]

    elif (ndim == 3):
        outpic = np.zeros((len(sigmavec),) + inpic.shape)
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :, :, :] = smoothpic[:, :, :]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :, :, :] = smoothpic[:, :, :]

    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)

    return outpic


def scaleoffset_variance(
        scspmethod : Union[str, ScSpMethod] = 'discgauss'
) -> float :
    """Returns the scale offset that the scale-space discretization method
scspmethod gives rise to at coarser scales. At finer scales, however, the
added offset may be lower.

Note that this scale offset is given in units of the variance s = sigma^2
of the kernel, as opposed to the standard deviation sigma, motivated by
the additive property of variances under convolution of non-negative kernels.
"""
    if (isinstance(scspmethod, str)):
        scspmethodname = scspmethod
    else:
        scspmethodname = scspmethod.methodname

    if (scspmethodname == 'discgauss'):
        scaleoffset = 0.0
    elif (scspmethodname == 'samplgauss'):
        scaleoffset = 0.0
    elif (scspmethodname == 'normsamplgauss'):
        scaleoffset = 0.0
    elif (scspmethodname == 'intgauss'):
        scaleoffset = 1/12        
    elif (scspmethodname == 'linintgauss'):
        scaleoffset = 1/6      
    else:
        raise ValueError('Scale space method %s not implemented' % scspmethodname)

    return scaleoffset


def discgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the discrete analogue of 
the Gaussian kernel with standard deviation sigma and relative truncation error 
less than epsilon.

Convolution with this kernel corresponds to solving a spatially discretized version
of the diffusion equation with the time variable = sigma^2 being continuous.

Over a 2-D spatial domain, the resulting scale-space representation obeys
non-enhancement of local extrema, meaning that the intensity value at any
spatial maximum is guaranteed to not increase with scale and that the 
intensity value at any spatial minimum is guaranteed to not decrease.

Over a 1-D spatial domain, the resulting scale-space representation does
also obey non-creation of local extrema, meaning that the number of local
extrema in the smoothed signal is guaranteed to not increase with scale.

The spatial standard deviation of the resulting kernel is exactly equal
to the scale parameter sigma over an infinite spatial domain. These kernel 
values do also in the ideal infinite case exactly sum up to one, and are
also confined in the interval [0, 1].

Reference: Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.
"""
    ndim = inpic.ndim
    sep1Dfilter = make1Ddiscgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = discgaussconv(inpic[:, :, layer], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Ddiscgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
) -> np.ndarray :
    """Generates a 1-D discrete analogue of the Gaussian kernel at scale level sigma
in units of the standard deviation of the kernel and with relative truncation error
not exceeding epsilon as a relative number over a D-dimensional spatial domain.
"""
    s = sigma*sigma
    tmpvecsize = ceil(1 + 1.5*gaussfiltsize(sigma, epsilon, D))
    # Generate filter coefficients from modified Bessel functions
    longhalffiltvec = ive(np.arange(0, tmpvecsize+1), s)
    halffiltvec = truncfilter(longhalffiltvec, truncerrtransf(epsilon, D))
    filtvec = mirrorhfilter(halffiltvec)
    return filtvec


def samplgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the sampled Gaussian 
kernel with standard deviation sigma and relative truncation error less than epsilon.

The transformation from the input image will always be a scale-space transformation,
in the sense that for a 1-D signal the number of local extrema in the smoothed
signal are guaranteed to not exceed the number of local extrema in the input image.
The transformation between adjacent scale levels is, however, not guaranteed to
be a scale-space transformation.

Note also that for smaller values of sigma, the kernel values may go outside the
interval [0, 1], which is not a desirable property.

For a theoretical explanations of these properties, see Section VII.A in

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

or Section 3.6.1 in

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
"""
    ndim = inpic.ndim
    sep1Dfilter = make1Dsamplgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = samplgaussconv(inpic[:, :, layer], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dsamplgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
) -> np.ndarray :
    """Generates a sampled Gaussian kernel with standard deviation sigma, given an
upper bound on the relative truncation error epsilon over a D-dimensional domain.
"""
    vecsize = ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    return gauss(x, sigma)


def gauss(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    return 1/(sqrt(2*pi)*sigma)*np.exp(-(x**2/(2*sigma**2)))


def normsamplgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the normalized 
sampled Gaussian kernel with standard deviation sigma and relative truncation 
error less than epsilon.

The transformation from the input image will always be a scale-space transformation,
in the sense that for a 1-D signal the number of local extrema in the smoothed
signal are guaranteed to not exceed the number of local extrema in the input image.
The transformation between adjacent scale levels is, however, not guaranteed to
be a scale-space transformation.

By a normalization of the discrete sampled Gaussian filter to unit l_1-norm,
this approach avoids the problems that the regular sampled Gaussian kernel
may assume values greater than 1 and the kernel values do not sum to 1.
The resulting filter kernel will, however, still be too narrow at very
fine scale, meaning that the normalization does not really solve the
real problems with the sampled Gaussian kernel at very fine scales.

For a theoretical explanations of the properties of the regular sampled
Gaussian kernel, see Section VII.A in

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

or Section 3.6.1 in

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
"""
    ndim = inpic.ndim
    sep1Dfilter = make1Dnormsamplgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = normsamplgaussconv(inpic[:, :, layer], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dnormsamplgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
) -> np.ndarray :
    """Generates a normalized sampled Gaussian kernel with standard deviation sigma, 
given an upper bound on the relative truncation error epsilon over a D-dimensional 
domain.
"""
    prelfilter = make1Dsamplgaussfilter(sigma, epsilon, D)
    return prelfilter/np.sum(prelfilter)


def intgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the integrated Gaussian 
kernel with standard deviation sigma and relative truncation error less than epsilon.

The transformation from the input image will always be a scale-space transformation,
in the sense that for a 1-D signal the number of local extrema in the smoothed
signal are guaranteed to not exceed the number of local extrema in the input image.
The transformation between adjacent scale levels is, however, not guaranteed to
be a scale-space transformation.

The kernel values of the resulting discrete approximation of the Gaussian kernel
do in the ideal infinite case exactly sum up to one, and are also confined in the 
interval [0, 1]. The spatial integration of the Gaussian kernel over the support
region of each pixel does, however, add a scale offset of 1/12 in units of the
variance equal to the squared standard deviation of the kernel at coarser scales.
This added variance corresponds to the spatial variance of a box filter over the 
support region of the image pixel over which the continuous Gaussian kernel is 
integrated.

For a theoretical explanation of these properties, see Section 3.6.3 in

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
"""
    ndim = inpic.ndim
    sep1Dfilter = make1Dintgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = intgaussconv(inpic[:, :, layer], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dintgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
) -> np.ndarray :
    """Generates an integrated Gaussian kernel with standard deviation sigma, given an
upper bound on the relative truncation error epsilon over a D-dimensional domain.

Remark: Adds additional spatial variance 1/12 to the kernel
"""
    vecsize = ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    return 1/2*(1 + erf(x/(sqrt(2)*sigma)))


def linintgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the linerily 
integrated Gaussian kernel with standard deviation sigma and relative 
truncation error less than epsilon.

The transformation from the input image will always be a scale-space transformation,
in the sense that for a 1-D signal the number of local extrema in the smoothed
signal are guaranteed to not exceed the number of local extrema in the input image.
The transformation between adjacent scale levels is, however, not guaranteed to
be a scale-space transformation.

The kernel values of the resulting discrete approximation of the Gaussian kernel
are confined in the interval [0, 1]. The spatial integration of the Gaussian kernel 
over the support region of each pixel does, however, add a scale offset of 1/6 in 
units of the variance equal to the squared standard deviation of the kernel at
coarser scales. This added variance corresponds to the spatial variance of a 
triangular extending to the neigbouring pixels, which is used as spatial window 
function when integrating the continuous Gaussian kernel. That triangular filter
does in turn correspond to the convolution of a box filter over a pixel
support region by itself, thus explaining the doubling of the scale offset
in relation to the scale offset for the integrated Gaussian kernel.

For a theoretical explanation of these properties, see Section 3.6.3 in

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
"""
    ndim = inpic.ndim
    sep1Dfilter = make1Dlinintgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = linintgaussconv(inpic[:, :, layer], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dlinintgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
) -> np.ndarray :
    """Generates a linearily integrated Gaussian kernel with standard deviation 
sigma, given an upper bound on the relative truncation error epsilon over a 
D-dimensional domain.

Remark: Adds additional spatial variance 1/6 to the kernel
"""
    vecsize = ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    # The following equation is the result of a closed form integration of
    # the expression for the filter coefficients in Eq (3.90) in Lindeberg (1993)
    # Scale-Space Theory in Computer Vision, Springer.
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + \
           x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - 2*gauss(x, sigma) + \
                       gauss(x - 1, sigma))


def x_scaled_erf(x : np.ndarray, sigma : float = 1.0):
    return x * scaled_erf(x, sigma)


def gaussfiltsize(sigma : float, epsND : float, D : int) -> float :
    """Estimates the necessary size to truncate a Gaussian kernel with 
standard deviation sigma to a relative truncation epsND over a D-dimensional 
domain.
"""
    s = sigma*sigma
    eps1D = truncerrtransf(epsND, D)
    N = sqrt(2*s)*erfcinv(eps1D)    
    return N


def truncerrtransf(epsND : float, D : int) -> float :
    """Converts a relative truncation error epsND over a D-dimensional domain to
a relative truncation error over a 1-D domain when using separable convolution.
"""
    eps1D = 1 - (1 - epsND)**(1/D)
    return eps1D


def truncfilter(longhalffilter : np.ndarray, epsilon : float) -> np.ndarray :
    """Truncates a filter with overestimated size to a more compact size, to save
computational work in the spatial convolutions that are to follow.
"""
    length = longhalffilter.shape[0]
    sum = longhalffilter[0]
    
    i = 1
    while ((sum < 1-epsilon) and (i < length)):
        sum = sum + 2*longhalffilter[i]
        i += 1

    return longhalffilter[0:i]


def mirrorhfilter(halffilter : np.ndarray) -> np.ndarray :
    """Extends a one-sided spatial filter to a symmetric filter by spatial mirroring.
"""
    length = halffilter.shape[0]
    revfilter = halffilter[::-1]
    return np.append(revfilter[0:length-1], halffilter)


def deltafcn(xsize : int, ysize : int) -> np.ndarray :
    """Generates a discrete delta function of size xsize x ysize pixels.
"""
    pic = np.zeros([xsize, ysize])

    if (xsize % 2):
        xc = round((xsize - 1)/2)
    else:
        xc = round(xsize/2)
    if (ysize % 2):
        yc = round((ysize - 1)/2)
    else:
        yc = round(ysize/2)

    pic[xc, yc] = 1.0

    return pic


def dxmask() -> np.ndarray :
    return np.array([[-1/2, 0, 1/2]])


def dymask() -> np.ndarray :
    return np.array([[+1/2], \
                     [   0], \
                     [-1/2]])


def dxxmask() -> np.ndarray :
    return np.array([[1, -2, 1]])


def dxymask() -> np.ndarray :
    return np.array([[-1/4, 0, +1/4], \
                     [   0, 0,    0], \
                     [+1/4, 0, -1/4]])


def dyymask() -> np.ndarray :
    return np.array([[+1], \
                     [-2], \
                     [+1]])


def dxxxmask() -> np.ndarray :
    return np.array([[-1/2, 1, 0, -1, 1/2]])


def dxxymask() -> np.ndarray :
    return np.array([[+1/2, -1, +1/2], \
                     [   0,  0,    0], \
                     [-1/2, +1, -1/2]])


def dxyymask() -> np.ndarray :
    return np.array([[-1/2, 0, +1/2], \
                     [  +1, 0,   -1], \
                     [-1/2, 0, +1/2]])


def dyyymask() -> np.ndarray :
    return np.array([[+1/2], \
                     [  -1], \
                     [   0], \
                     [  +1], \
                     [-1/2]])


def dxxxxmask() -> np.ndarray :
    return np.array([[1, -4, 6, -4, 1]])


def dxxxymask() -> np.ndarray :
    return np.array([[-1/4, +1/2, 0, -1/2, +1/4], \
                     [   0,    0, 0,    0,    0], \
                     [+1/4, -1/2, 0, +1/2, -1/4]])


def dxxyymask() -> np.ndarray :
    return np.array([[+1, -2, +1], \
                     [-2, +4, -2], \
                     [+1, -2, +1]])


def dxyyymask() -> np.ndarray :
    return np.array([[-1/4, 0, +1/4], \
                     [+1/2, 0, -1/2], \
                     [   0, 0,    0], \
                     [-1/2, 0, +1/2], \
                     [+1/4, 0, -1/4]])


def dyyyymask() -> np.ndarray :
    return np.array([[+1], \
                     [-4], \
                     [+6], \
                     [-4], \
                     [+1]])


def computeNjetfcn(
        inpic,
        njetfcn : str,
        sigma : float,
        normdermethod : Union[str, ScSpNormDerMethod] = 'discgaussLp'
) -> np.ndarray :
    """Computes an N-jet function in terms of scale-normalized Gaussian derivatives 
of the image inpic at scale level sigma in units of the standard deviation of
the Gaussian kernel, and using the scale normalization method normdermethod.

Implemented N-jet functions:
  'L' - smoothed scale-space representation
  'Lx' - 1:st-order partial derivative in x-direction
  'Ly' - 1:st-order partial derivative in y-direction
  'Lxx' - 2:nd-order partial derivative in x-direction
  'Lxy' - mixed 2nd-order partial derivative in x- and y-directions
  'Lyy' - 2:nd-order partial derivative in y-direction
  'Lv' - gradient magnitude
  'Lv2' - squared gradient magnitude
  'Laplace' - Laplacian operator
  'detHessian' - determinant of the Hessian
  'sqrtdetHessian' - signed square root of absolute value of determinant of the Hessian
  'Kappa' - rescaled level curve curvature
  'Lv2Lvv' - 2:nd-order directional derivative in gradient direction * Lv^2
  'Lv3Lvvv' - 3:rd-order directional derivative in gradient direction * Lv^3
  'Lp' - 1:st-order directional derivative in 1:st principal curvature direction
  'Lq' - 1:st-order directional derivative in 2:nd principal curvature direction
  'Lpp' - 2:nd-order directional derivative in 1:st principal curvature direction
  'Lqq' - 2:nd-order directional derivative in 2:nd principal curvature direction
In addition, 3:rd- and 4:th-order partial derivatives are also implemented.

The differential expressions 'Lv', 'Lv2', 'Lv2Lvv' and 'Lv3Lvvv' are used in
methods for edge detection. The differential expressions 'Laplace', 'detHessian'
and 'sqrtdetHessian' are used in methods for interest point detection, 
blob detection and corner detection. The differential expressions 'Lp', 'Lq',
'Lpp' and 'Lqq' are used in methods for ridge detection.

This function is the preferred choice for simplicitly or if you only need a single 
N-jetfcn at the given scale. If you instead want to compute multiple N-jetfcns
at the same scale, it is, however, computationally more efficient to perform
the scale-space smoothing yourself using the scspconv() function and then
applying the function applyNjetfcn() multiple times for each N-jetfcn.
"""
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    smoothpic = scspconv(inpic, sigma, normdermethod.scspmethod)
    return applyNjetfcn(smoothpic, njetfcn, sigma, normdermethod)


def applyNjetfcn(
        smoothpic : np.ndarray,
        njetfcn : str,
        sigma : float = 1.0,
        normdermethod : Union[str, ScSpNormDerMethod] = 'discgaussLp'
) -> np.ndarray :
    """Applies an N-jet function in terms of scale-normalized Gaussian derivatives 
to an already computed scale-space representation at scale level sigma in units
of the standard deviation of the Gaussian kernel, and using the scale normalization
method normdermethod.

Implemented N-jet functions:
  'L' - smoothed scale-space representation
  'Lx' - 1:st-order partial derivative in x-direction
  'Ly' - 1:st-order partial derivative in y-direction
  'Lxx' - 2:nd-order partial derivative in x-direction
  'Lxy' - mixed 2nd-order partial derivative in x- and y-directions
  'Lyy' - 2:nd-order partial derivative in y-direction
  'Lv' - gradient magnitude
  'Lv2' - squared gradient magnitude
  'Laplace' - Laplacian operator
  'detHessian' - determinant of the Hessian
  'sqrtdetHessian' - signed square root of absolute value of determinant of the Hessian
  'Kappa' - rescaled level curve curvature
  'Lv2Lvv' - 2:nd-order directional derivative in gradient direction * Lv^2
  'Lv3Lvvv' - 3:rd-order directional derivative in gradient direction * Lv^3
  'Lp' - 1:st-order directional derivative in 1:st principal curvature direction
  'Lq' - 1:st-order directional derivative in 2:nd principal curvature direction
  'Lpp' - 2:nd-order directional derivative in 1:st principal curvature direction
  'Lqq' - 2:nd-order directional derivative in 2:nd principal curvature direction

The differential expressions 'Lv', 'Lv2', 'Lv2Lvv' and 'Lv3Lvvv' are used in
methods for edge detection. The differential expressions 'Laplace', 'detHessian'
and 'sqrtdetHessian' are used in methods for interest point detection, 
blob detection and corner detection. The differential expressions 'Lp', 'Lq',
'Lpp' and 'Lqq' are used in methods for ridge detection.
"""
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)
        
    if ((smoothpic.ndim == 3) and (smoothpic.shape[2] > 1)):
        numlayers = smoothpic.shape[2]
        outpic = np.zeros(smoothpic.shape)
        for layer in range(0, numlayers):
            outpic[:, :, layer] = \
              applyNjetfcn(smoothpic[:, :, layer], njetfcn, sigma, normdermethod)
    else:
        if (njetfcn == 'L'):
            outpic = smoothpic
        elif (njetfcn == 'Lx'):
            outpic = normderfactor(1, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxmask())
        elif (njetfcn == 'Ly'):
            outpic = normderfactor(0, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dymask())
        elif (njetfcn == 'Lxx'):
            outpic = normderfactor(2, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxmask())
        elif (njetfcn == 'Lxy'):
            outpic = normderfactor(1, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxymask())
        elif (njetfcn == 'Lyy'):
            outpic = normderfactor(0, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dyymask())
        elif (njetfcn == 'Lv'):
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())   
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            outpic = np.sqrt(Lx*Lx + Ly*Ly)
        elif (njetfcn == 'Lv2'):
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())   
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            outpic = Lx*Lx + Ly*Ly
        elif (njetfcn == 'Laplace'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())   
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Lxx + Lyy
        elif (njetfcn == 'detHessian'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Lxx*Lyy - Lxy*Lxy
        elif (njetfcn == 'sqrtdetHessian'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            detHessian = Lxx*Lyy - Lxy*Lxy
            outpic = np.sign(detHessian) * np.sqrt(np.abs(detHessian))
        elif (njetfcn == 'Kappa'):
            # Rescaled level curve curvature
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Ly*Ly*Lxx + Lx*Lx*Lyy - 2*Lx*Ly*Lxy
        elif (njetfcn == 'Lv2Lvv'):
            # 2nd-order derivative in gradient direction (used for edge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Lx*Lx*Lxx + 2*Lx*Ly*Lxy + Ly*Ly*Lyy
        elif (njetfcn == 'Lv3Lvvv'):
            # 3rd-order derivative in gradient direction (used for edge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            Lxxx = normderfactor(3, 0, sigma, normdermethod) * \
                   correlate(smoothpic, dxxxmask())
            Lxxy = normderfactor(2, 1, sigma, normdermethod) * \
                   correlate(smoothpic, dxxymask())
            Lxyy = normderfactor(1, 2, sigma, normdermethod) * \
                   correlate(smoothpic, dxyymask())
            Lyyy = normderfactor(0, 3, sigma, normdermethod) * \
                   correlate(smoothpic, dyyymask())
            outpic = Lx*Lx*Lx*Lxxx + 3*Lx*Lx*Ly*Lxxy + \
                     3*Lx*Ly*Ly*Lxyy + Ly*Ly*Ly*Lyyy
        elif (njetfcn == 'Lp'):
            # 1st-order derivative in first principal curvature direction
            # (used for ridge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + \
                                np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = sinbeta * Lx - cosbeta * Ly
        elif (njetfcn == 'Lq'):
            # 1st-order derivative in second principal curvature
            # (used for valley detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + \
                                np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = cosbeta * Lx + sinbeta * Ly
        elif (njetfcn == 'Lpp'):
            # 2nd-order derivative in first principal curvature direction
            # (used for ridge detection)
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = ((Lxx + Lyy) - \
                      np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2
        elif (njetfcn == 'Lqq'):
            # 2nd-order derivative in second principal curvature direction
            # (used for valley detection)
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = ((Lxx + Lyy) + \
                      np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2
        elif (njetfcn == 'Lxxx'):
            outpic = normderfactor(3, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxmask())
        elif (njetfcn == 'Lxxy'):
            outpic = normderfactor(2, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxxymask())
        elif (njetfcn == 'Lxyy'):
            outpic = normderfactor(1, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dxyymask())
        elif (njetfcn == 'Lyyy'):
            outpic = normderfactor(0, 3, sigma, normdermethod) * \
                     correlate(smoothpic, dyyymask())
        elif (njetfcn == 'Lxxxx'):
            outpic = normderfactor(4, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxxmask())
        elif (njetfcn == 'Lxxxy'):
            outpic = normderfactor(3, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxymask())
        elif (njetfcn == 'Lxxyy'):
            outpic = normderfactor(2, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dxxyymask())
        elif (njetfcn == 'Lxyyy'):
            outpic = normderfactor(1, 3, sigma, normdermethod) * \
                     correlate(smoothpic, dxyyymask())
        elif (njetfcn == 'Lyyyy'):
            outpic = normderfactor(0, 4, sigma, normdermethod) * \
                     correlate(smoothpic, dyyyymask())
        else:
            raise ValueError('NJetFcn %s not implemented yet' % njetfcn)

    return outpic


def normderfactor(
        xorder : int,
        yorder : int,
        sigma : float,
        normdermethod : Union[str, ScSpNormDerMethod]
) -> float :
    """Compute the scale normalization factor for the scale-normalied 
Gaussian derivative of order xorder in the x-direction and of order yorder 
in the y-direction at scale sigma in units of the standard deviation of 
the Gaussian kernel and using the scale normalization method normdermethod.
"""
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    if (normdermethod.normdermethod == 'nonormalization'):
        value = 1.0
    elif (normdermethod.normdermethod == 'varnorm'):
        # ==>> Here it could be natural to compensate for the additional variance
        # ==>> for the integrated or linearly integrated Gaussian kernels.
        value = sigma**(xorder + yorder)
    elif (normdermethod.normdermethod == 'Lpnorm'):
        if (normdermethod.scspmethod.methodname == 'discgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (discgaussder1D_L1norm(xorder, sigma, \
                                      normdermethod.scspmethod.epsilon) * \
                discgaussder1D_L1norm(yorder, sigma, \
                                      normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented \
for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'samplgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (samplgaussder1D_L1norm(xorder, sigma, \
                                        normdermethod.scspmethod.epsilon) * \
                samplgaussder1D_L1norm(yorder, sigma, \
                                       normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented \
for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'normsamplgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (normsamplgaussder1D_L1norm(xorder, sigma, \
                                            normdermethod.scspmethod.epsilon) * \
                 normsamplgaussder1D_L1norm(yorder, sigma, \
                                            normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented \
for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'intgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (intgaussder1D_L1norm(xorder, sigma, \
                                      normdermethod.scspmethod.epsilon) * \
                intgaussder1D_L1norm(yorder, sigma, \
                                     normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented \
for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'linintgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (linintgaussder1D_L1norm(xorder, sigma, \
                                         normdermethod.scspmethod.epsilon) * \
                linintgaussder1D_L1norm(yorder, sigma, \
                                        normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented \
for gamma = 1.0')
        else:
            raise ValueError('Lp-normalization not implemented for \
scale-space derivative method %s' % normdermethod.scspmethod.methodname)
    else:
        raise ValueError('Derivative method %s not implemented yet' \
                         % normdermethod.normdermethod)

    return value


def normgaussder1D_L1norm(
        order : int,
        sigma : float,
        gammapar : float = 1.0
) -> float :
    """Returns the L_1-norm of 1-D gamma-normalized Gaussian derivative 
of order sigma and at scale sigma in units of the standard deviation of 
the Gaussian kernel, using the scale normalization parameter gammapar.
"""
    s = sigma*sigma
    
    if (order == 0):
        value = 1.0
    elif (order == 1):
        value = sqrt(2/pi) * s**((gammapar - 1)/2)
    elif (order == 2):
        value = 2*sqrt(2/(exp(1)*pi)) * s**(gammapar - 1)
    elif (order == 3):
        value = (4 + exp(3/2))*sqrt(2/pi)/exp(3/2) * s**(3*(gammapar - 1)/2)
    elif (order == 4):
        value = \
          2*exp(-3/2 - sqrt(3/2)) * \
          (2*exp(sqrt(6))*sqrt((9 - 3*sqrt(6))/pi) \
               + 3*sqrt((3 - sqrt(6))/pi) \
          + sqrt(6*(3 - sqrt(6))/pi) \
          + sqrt(3*(3 + sqrt(6))/pi)) * s**(2*(gammapar - 1))
    else:
        raise ValueError('Not implemented for order %d' % order)

    return value


def discgaussder1D_L1norm(
        order : int,
        sigma : float,
        epsilon : float = 0.00000001
) -> float :
    """Returns the L_1-norm of the n:th-order difference of the 1-D discrete analogue 
of the Gaussian kernel at scale level sigma in units of the standard deviation and 
truncated at the tails with a relative approximation error less than epsilon.
"""
    smoothkernel = make1Ddiscgaussfilter(sigma, epsilon, 1)

    if (order == 0):
        derkernel = smoothkernel
    elif (order == 1):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 0, 1/2]))
    elif (order == 2):
        derkernel = correlate1d(smoothkernel, np.array([1, -2, 1]))
    elif (order == 3):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 1, 0, -1, 1/2]))
    elif (order == 4):
        derkernel = correlate1d(smoothkernel, np.array([1, -4, 6, -4, 1]))
    else:
        raise ValueError('Not implemented for order %d yet' % order)

    return sum(abs(derkernel))


def samplgaussder1D_L1norm(
        order : int,
        sigma : float,
        epsilon : float = 0.00000001
) -> float :
    """Returns the L_1-norm of the n:th-order difference of the 1-D sampled 
Gaussian kernel at scale level sigma in units of the standard deviation and 
truncated at the tails with a relative approximation error less than epsilon.
"""
    smoothkernel = make1Dsamplgaussfilter(sigma, epsilon, 1)

    if (order == 0):
        derkernel = smoothkernel
    elif (order == 1):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 0, 1/2]))
    elif (order == 2):
        derkernel = correlate1d(smoothkernel, np.array([1, -2, 1]))
    elif (order == 3):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 1, 0, -1, 1/2]))
    elif (order == 4):
        derkernel = correlate1d(smoothkernel, np.array([1, -4, 6, -4, 1]))
    else:
        raise ValueError('Not implemented for order %d yet' % order)

    return sum(abs(derkernel))


def normsamplgaussder1D_L1norm(
        order : int,
        sigma : float,
        epsilon : float = 0.00000001
) -> float :
    """Returns the L_1-norm of the n:th-order difference of the 1-D sampled 
Gaussian kernel at scale level sigma in units of the standard deviation and 
truncated at the tails with a relative approximation error less than epsilon.
"""
    smoothkernel = make1Dnormsamplgaussfilter(sigma, epsilon, 1)

    if (order == 0):
        derkernel = smoothkernel
    elif (order == 1):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 0, 1/2]))
    elif (order == 2):
        derkernel = correlate1d(smoothkernel, np.array([1, -2, 1]))
    elif (order == 3):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 1, 0, -1, 1/2]))
    elif (order == 4):
        derkernel = correlate1d(smoothkernel, np.array([1, -4, 6, -4, 1]))
    else:
        raise ValueError('Not implemented for order %d yet' % order)

    return sum(abs(derkernel))


def intgaussder1D_L1norm(
        order : int,
        sigma : float,
        epsilon : float = 0.00000001
) -> float :
    """Returns the L_1-norm of the n:th-order difference of the 1-D integrated 
Gaussian kernel at scale level sigma in units of the standard deviation and 
truncated at the tails with a relative approximation error less than epsilon.
"""
    smoothkernel = make1Dintgaussfilter(sigma, epsilon, 1)

    if (order == 0):
        derkernel = smoothkernel
    elif (order == 1):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 0, 1/2]))
    elif (order == 2):
        derkernel = correlate1d(smoothkernel, np.array([1, -2, 1]))
    elif (order == 3):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 1, 0, -1, 1/2]))
    elif (order == 4):
        derkernel = correlate1d(smoothkernel, np.array([1, -4, 6, -4, 1]))
    else:
        raise ValueError('Not implemented for order %d yet' % order)

    return sum(abs(derkernel))


def linintgaussder1D_L1norm(
        order : int,
        sigma : float,
        epsilon : float = 0.00000001
) -> float :
    """Returns the L_1-norm of the n:th-order difference of the 1-D linearly 
integrated Gaussian kernel at scale level sigma in units of the standard deviation 
and truncated at the tails with a relative approximation error less than epsilon.
"""
    smoothkernel = make1Dlinintgaussfilter(sigma, epsilon, 1)

    if (order == 0):
        derkernel = smoothkernel
    elif (order == 1):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 0, 1/2]))
    elif (order == 2):
        derkernel = correlate1d(smoothkernel, np.array([1, -2, 1]))
    elif (order == 3):
        derkernel = correlate1d(smoothkernel, np.array([-1/2, 1, 0, -1, 1/2]))
    elif (order == 4):
        derkernel = correlate1d(smoothkernel, np.array([1, -4, 6, -4, 1]))
    else:
        raise ValueError('Not implemented for order %d yet' % order)

    return sum(abs(derkernel))


def defaultscspnormdermethodobject(
        scspnormdermethod : str = 'discgaussLp',
        gamma : float = 1.0
) -> ScSpNormDerMethod :
    """Converts a user-friendly string for a method for approximating 
scale-normalized derivatives for discrete data to a class object, including 
also a specification of the discretization method to use for discrete 
approximation of the underlying spatial smoothing operation
"""
    if (scspnormdermethod == 'discgauss'):
        object = scspnormdermethodobject('discgauss', 'nonormalization', gamma)
    elif (scspnormdermethod == 'discgaussvar'):
        object = scspnormdermethodobject('discgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'discgaussLp'):
        object = scspnormdermethodobject('discgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'samplgauss'):
        object = scspnormdermethodobject('samplgauss', 'nonormalization', gamma)
    elif (scspnormdermethod == 'samplgaussvar'):
        object = scspnormdermethodobject('samplgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'samplgaussLp'):
        object = scspnormdermethodobject('samplgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'normsamplgauss'):
        object = scspnormdermethodobject('normsamplgauss', 'nonormalization', gamma)
    elif (scspnormdermethod == 'normsamplgaussvar'):
        object = scspnormdermethodobject('normsamplgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'normsamplgaussLp'):
        object = scspnormdermethodobject('normsamplgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'intgauss'):
        object = scspnormdermethodobject('intgauss', 'nonormalization', gamma)
    elif (scspnormdermethod == 'intgaussvar'):
        object = scspnormdermethodobject('intgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'intgaussLp'):
        object = scspnormdermethodobject('intgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'linintgauss'):
        object = scspnormdermethodobject('linintgauss', 'nonormalization', gamma)
    elif (scspnormdermethod == 'linintgaussvar'):
        object = scspnormdermethodobject('linintgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'linintgaussLp'):
        object = scspnormdermethodobject('linintgauss', 'Lpnorm', gamma)
    else:
        raise ValueError('Scale-space derivative method %s not implemented yet',
                             scspnormdermethod)
    return object


def variance(filter : np.ndarray) -> np.ndarray:
    """Returns the spatial covariance matrix of 2-D filter, assumed to be non-negative.
"""
    if (filter.ndim != 2):
        raise ValueError('Only implemented for 2-D filters so far')

    xsize = filter.shape[1]
    ysize = filter.shape[0]
    if (xsize % 2):
        x = np.linspace(-(xsize-1)/2, (xsize-1)/2, xsize)
    else:
        # Choose convention to fit deltafcn()
        x = np.linspace(-xsize/2, xsize/2-1, xsize)
    if (ysize % 2):
        y = np.linspace(-(ysize-1)/2, (ysize-1)/2, ysize)
    else:
        # Choose convention to fit deltafcn()
        y = np.linspace(-ysize/2, ysize/2-1, ysize)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    x2mom = np.sum(np.sum(xv * xv * filter))/np.sum(np.sum(filter))
    xymom = np.sum(np.sum(xv * yv * filter))/np.sum(np.sum(filter))
    y2mom = np.sum(np.sum(yv * yv * filter))/np.sum(np.sum(filter))

    xmean, ymean = filtermean(filter)

    return np.array([[x2mom - xmean*xmean, xymom - xmean*ymean], \
                     [xymom - xmean*ymean, y2mom - ymean*ymean]])


def filtermean(filter : np.ndarray) -> (float, float) :
    """Returns the spatial mean vector of 2-D filter, assumed to be non-negative.
"""
    if (filter.ndim != 2):
        raise ValueError('Only implemented for 2-D filters so far')

    xsize = filter.shape[1]
    ysize = filter.shape[0]
    if (xsize % 2):
        x = np.linspace(-(xsize-1)/2, (xsize-1)/2, xsize)
    else:
        # Choose convention to fit deltafcn()
        x = np.linspace(-xsize/2, xsize/2-1, xsize)
    if (ysize % 2):
        y = np.linspace(-(ysize-1)/2, (ysize-1)/2, ysize)
    else:
        # Choose convention to fit deltafcn()
        y = np.linspace(-ysize/2, ysize/2-1, ysize)
    xv, yv = np.meshgrid(x, y, indexing='xy')

    xmean = np.sum(np.sum(xv * filter))/np.sum(np.sum(filter))
    ymean = np.sum(np.sum(yv * filter))/np.sum(np.sum(filter))
    
    return xmean, ymean


def mean1D(filter : np.ndarray) -> float:
    """Computes the spatial mean of a non-negative filter.
"""
    
    if filter.ndim != 1:
        raise ValueError('Only implemented for 1-D filters')

    size = filter.shape[0]
    x = np.linspace(0, size-1, size)

    return np.sum(np.sum(x * filter)) / np.sum(np.sum(filter))


def variance1D(filter : np.ndarray) -> float:
    """Computes the spatial variance of a non-negative filter.
"""
    
    if filter.ndim != 1:
        raise ValueError('Only implemented for 1-D filters')

    size = filter.shape[0]
    x = np.linspace(0, size-1, size)

    x2mom = np.sum(np.sum(x * x * filter)) / np.sum(np.sum(filter))
    xmean = mean1D(filter)

    return x2mom - xmean * xmean


def RGB2LUV(inpic) -> np.ndarray:
    """Converts an RGB colour image to a colour-opponent LUV colour space.
"""
    inpic = np.array(inpic).astype('float')
    outpic = np.zeros(inpic.shape)
    outpic[:, :, 0] = (inpic[:, :, 0] + inpic[:, :, 1] + inpic[:, :, 2])/3.0
    outpic[:, :, 1] = 1.0*(inpic[:, :, 0] - inpic[:, :, 1])
    outpic[:, :, 2] = (inpic[:, :, 0] + inpic[:, :, 1])/2.0 - inpic[:, :, 2]
    return(outpic)


def RGB2L(inpic) -> np.ndarray:
    """Converts an RGB colour image to a greylevel image
"""
    inpic = np.array(inpic).astype('float')
    return((inpic[:, :, 0] + inpic[:, :, 1] + inpic[:, :, 2])/3.0)


