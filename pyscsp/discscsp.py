"""Discrete Scale Space and Scale-Space Derivative Toolbox for Python

For computing discrete scale-space smoothing by convolution with the discrete
analogue of the Gaussian kernel and for computing discrete derivative approximations
by applying central difference operators to the smoothed data. Then, different
types of feature detectors can be defined, by combining discrete analogues of the
Gaussian derivative operators into differential expressions.

This code is the result of porting a subset of the routines in the Matlab packages
discscsp and discscspders to Python, however, with different interfaces for some
of the functions.

Note: The scale normalization does not explicitly compensate for the additional 
variance 1/12 for the integrated Gaussian kernel or the additional variance 1/6
for the linearly integrated Gaussian kernel at coarser scales.

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

Lindeberg (2015) "Image matching using generalized scale-space interest points", 
Journal of Mathematical Imaging and Vision, 52(1): 3-36.

Compared to the original Matlab code, the following implementation is reduced 
in the following ways:

- there is no handling of scale normalization powers gamma that are not equal to one,
- much fewer functions of the N-jet have so far been implemented,
- there is no passing of additional parameters to functions of the N-jet,
- this reimplementation has not been thoroughly tested.
"""
from math import sqrt, exp, ceil, pi, cos, sin
from typing import NamedTuple, Union, List
import numpy as np
from scipy.ndimage import correlate1d, correlate
from scipy.special import ive, erf, erfcinv


class ScSpMethod(NamedTuple):
    """Object for storing the parameters of a scale-space discretization method, 
    which can be of either of the types  'discgauss', 'samplgauss', 'normsamplgauss', 
    'intgauss' or 'linintgauss'
    """
    methodname: str
    epsilon: float


class ScSpNormDerMethod(NamedTuple):
    """Object for storing the parameters of a discretization method for computing
    scale-normalized derivatives, including also the necessary parameters of the
    underlying method for discrete approximation of the Gaussian smoothing operation.
    """
    scspmethod: ScSpMethod
    normdermethod: str # either 'nonormalization' or 'varnorm' 
    gamma: float


def scspnormdermethodobject(
        scspmethod : str = 'discgauss',
        normdermethod : str = 'varnorm',
        gamma : float = 1.0,
        epsilon : float = 0.00000001
    ) -> ScSpNormDerMethod :
    """Creates an object that contains the parameters of discretization method
    for computing scale-normalized derivatives, with default values for a preferred 
    choice.
    """
    return ScSpNormDerMethod(ScSpMethod(scspmethod, epsilon), normdermethod, gamma)


def scspconv(
        inpic,
        sigma : float,
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.ndarray :
    """Computes the scale-space representation of the 2-D image inpic (or a 1-D signal) 
    at scale level sigma in units of the standard deviation of the Gaussian kernel, 
    that is approximated discretely with the method scspmethod, and with the formally 
    infinite convolution operation truncated at the tails with a relative approximation 
    error less than epsilon.

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
    if isinstance(scspmethod, str):
        scspmethodname = scspmethod
    else:
        scspmethodname = scspmethod.methodname
        epsilon = scspmethod.epsilon

    if scspmethodname == 'discgauss':
        outpic = discgaussconv(inpic, sigma, epsilon)

    elif scspmethodname == 'samplgauss':
        outpic = samplgaussconv(inpic, sigma, epsilon)

    elif scspmethodname == 'normsamplgauss':
        outpic = normsamplgaussconv(inpic, sigma, epsilon)

    elif scspmethodname == 'intgauss':
        outpic = intgaussconv(inpic, sigma, epsilon)

    elif scspmethodname == 'linintgauss':
        outpic = linintgaussconv(inpic, sigma, epsilon)

    else:
        raise ValueError(f'Scale space method {scspmethodname} not implemented')

    return outpic


def scspconv_mult(
        inpic,
        sigmavec : List[float],
        scspmethod : Union[str, ScSpMethod] = 'discgauss',
        epsilon : float = 0.00000001
    ) -> np.ndarray :
    """Performs a similar function as the function scscpconv, with the 
    difference that an array of sigma values can be provided instead of a 
    single value, and that the cascade smoothing property of Gaussian 
    convolution is then made use of to perform the computational more
    efficiently than if applying the scspconv function for each
    scale level separately.

    Note, however, that the cascade smoothing property works best when
    using the discrete analogue of the Gaussian kernel 'discgauss', for
    which the cascade smoothing property holds exactly in the ideal
    case of an infinite image domain with kernels the smoothing kernels
    having infinite support. For the other ways of approximating the
    Gaussian kernel discretely, there may be deviations depending on
    the scale values.

    The output will be an array of one dimension more than for the 
    function scspconv.

    Note that the scale values in the list sigmavec must be in
    increasing order.
    """
    ndim = inpic.ndim

    if ndim == 1:
        outpic = np.zeros((len(sigmavec), len(inpic)))
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :] = smoothpic[:]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :] = smoothpic[:]

    elif ndim == 2:
        outpic = np.zeros((len(sigmavec),) + inpic.shape)
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :, :] = smoothpic[:, :]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :, :] = smoothpic[:, :]

    elif ndim == 3:
        outpic = np.zeros((len(sigmavec),) + inpic.shape)
        smoothpic = scspconv(inpic, sigmavec[0], scspmethod, epsilon)
        outpic[0, :, :, :] = smoothpic[:, :, :]
        for i in range(1, len(sigmavec)):
            sigmainc = sqrt(sigmavec[i] ** 2 - sigmavec[i-1] ** 2)
            smoothpic = scspconv(smoothpic, sigmainc, scspmethod, epsilon)
            outpic[i, :, :, :] = smoothpic[:, :, :]

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

    return outpic


def scaleoffset_variance(
        scspmethod : Union[str, ScSpMethod] = 'discgauss'
    ) -> float :
    """Returns the scale offset that the scale-space discretization method
    scspmethod gives rise to at coarser scales. At finer scales, however, 
    the added offset may be lower.

    Note that this scale offset is given in units of the variance s = sigma^2
    of the kernel, as opposed to the standard deviation sigma, motivated by
    the additive property of variances under convolution of non-negative kernels.
    """
    if isinstance(scspmethod, str):
        scspmethodname = scspmethod
    else:
        scspmethodname = scspmethod.methodname

    if scspmethodname == 'discgauss':
        scaleoffset = 0.0
    elif scspmethodname == 'samplgauss':
        scaleoffset = 0.0
    elif scspmethodname == 'normsamplgauss':
        scaleoffset = 0.0
    elif scspmethodname == 'intgauss':
        scaleoffset = 1/12
    elif scspmethodname == 'linintgauss':
        scaleoffset = 1/6
    else:
        raise ValueError(f'Scale space method {scspmethodname} not implemented')

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

    if ndim == 1:
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)

    elif ndim == 2:
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)

    elif ndim == 3:
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = discgaussconv(inpic[:, :, layer], sigma, epsilon)

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

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

    # Generate filter coefficients from the modified Bessel functions
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
    kernel with standard deviation sigma and relative truncation error less than 
    epsilon.

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

    if ndim == 1:
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)

    elif ndim == 2:
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)

    elif ndim == 3:
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = samplgaussconv(inpic[:, :, layer], sigma, epsilon)

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

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
    """Computes a Gaussian function given a set of spatial x-coordinates and sigma
    value specifying the standard deviation of the kernel.
    """
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

    if ndim == 1:
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)

    elif ndim == 2:
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)

    elif ndim == 3:
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = normsamplgaussconv(inpic[:, :, layer], sigma, epsilon)

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

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
    """Convolves the 2-D image inpic (or a 1-D signal) with the box integrated 
    Gaussian kernel with standard deviation sigma and relative truncation error less 
    than epsilon, according to Equation (3.89) on page 97 in Lindeberg (1993) 
    Scale-Space Theory  in Computer Vision, published by Springer.

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

    if ndim == 1:
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)

    elif ndim == 2:
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)

    elif ndim == 3:
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = intgaussconv(inpic[:, :, layer], sigma, epsilon)

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

    return outpic


def make1Dintgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
    ) -> np.ndarray :
    """Generates a box integrated Gaussian kernel with standard deviation sigma, 
    according to Equation (3.89) on page 97 in Lindeberg (1993) Scale-Space Theory 
    in Computer Vision (published by Springer), given an upper bound on the 
    relative truncation error epsilon over a D-dimensional domain.

    Remark: Adds additional spatial variance 1/12 to the kernel at coarser scales.
    """
    vecsize = ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)

    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes the scaled error function (as depending on a scale parameter sigma)
    given an array of x-coordinates.
    """
    return 1/2*(1 + erf(x/(sqrt(2)*sigma)))


def linintgaussconv(
        inpic,
        sigma : float,
        epsilon : float = 0.00000001
    ) -> np.ndarray :
    """Convolves the 2-D image inpic (or a 1-D signal) with the linearily 
    integrated Gaussian kernel with standard deviation sigma and relative 
    truncation error less than epsilon, according to Equation (3.89) on 
    page 97 in Lindeberg (1993) Scale-Space Theory  in Computer Vision, 
    published by Springer.

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

    if ndim == 1:
        outpic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter)

    elif ndim == 2:
        tmppic = correlate1d(np.array(inpic).astype('float'), sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)

    elif ndim == 3:
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for layer in range(0, inpic.shape[2]):
            outpic[:, :, layer] = linintgaussconv(inpic[:, :, layer], sigma, epsilon)

    else:
        raise ValueError(f'Cannot handle images of dimensionality {ndim}')

    return outpic


def make1Dlinintgaussfilter(
        sigma : float,
        epsilon : float = 0.00000001,
        D : int = 1
    ) -> np.ndarray :
    """Generates a linearily integrated Gaussian kernel with standard deviation 
    sigma, given an upper bound on the relative truncation error epsilon over a 
    D-dimensional domain.

    Remark: Adds additional spatial variance 1/6 to the kernel at coarser scales.
    """
    vecsize = ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)

    # The following equation is the result of a closed form integration of
    # the expression for the filter coefficients in Eq (3.90) on page 97
    # in Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + \
           x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - 2*gauss(x, sigma) + \
                       gauss(x - 1, sigma))


def x_scaled_erf(x : np.ndarray, sigma : float = 1.0):
    """Computes the product of the x-coordinate and the scaled error 
    function (as depending on a scale parameter sigma) given an array of x-coordinates.
    """
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
    filtersum = longhalffilter[0]

    i = 1
    while ((filtersum < 1-epsilon) and (i < length)):
        filtersum = filtersum + 2*longhalffilter[i]
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

    if xsize % 2:
        xc = round((xsize - 1)/2)
    else:
        xc = round(xsize/2)

    if ysize % 2:
        yc = round((ysize - 1)/2)
    else:
        yc = round(ysize/2)

    pic[xc, yc] = 1.0

    return pic


def dxmask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the first-order 
    derivative in the x-direction.
    """
    return np.array([[-1/2, 0, +1/2]])


def dymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the first-order 
    derivative in the y-direction.
    """
    return np.array([[+1/2], \
                     [   0], \
                     [-1/2]])


def dxxmask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the second-order 
    derivative in the x-direction.
    """
    return np.array([[1, -2, 1]])


def dxymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed second-order 
    derivative in the x- and y-directions.
    """
    return np.array([[-1/4, 0, +1/4], \
                     [   0, 0,    0], \
                     [+1/4, 0, -1/4]])


def dyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the second-order 
    derivative in the y-direction.
    """
    return np.array([[+1], \
                     [-2], \
                     [+1]])


def dxxxmask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the third-order 
    derivative in the x-direction.
    """
    return np.array([[-1/2, +1, 0, -1, 1/2]])


def dxxymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed third-order 
    derivative corresponding to a second-order derivative in the x-direction 
    and a first-order derivative in the y-direction.
    """
    return np.array([[+1/2, -1, +1/2], \
                     [   0,  0,    0], \
                     [-1/2, +1, -1/2]])


def dxyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed third-order 
    derivative corresponding to a first-order derivative in the x-direction 
    and a second-order derivative in the y-direction.
    """
    return np.array([[-1/2, 0, +1/2], \
                     [  +1, 0,   -1], \
                     [-1/2, 0, +1/2]])


def dyyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the third-order 
    derivative in the y-direction.
    """
    return np.array([[+1/2], \
                     [  -1], \
                     [   0], \
                     [  +1], \
                     [-1/2]])


def dxxxxmask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the fourth-order 
    derivative in the x-direction.
    """
    return np.array([[1, -4, 6, -4, 1]])


def dxxxymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed fourth-order 
    derivative corresponding to a third-order derivative in the x-direction and 
    a first-order derivative in the y-direction.
    """
    return np.array([[-1/4, +1/2, 0, -1/2, +1/4], \
                     [   0,    0, 0,    0,    0], \
                     [+1/4, -1/2, 0, +1/2, -1/4]])


def dxxyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed fourth-order 
    derivative corresponding to a second-order derivatives in the x- and y-directions.
    """
    return np.array([[+1, -2, +1], \
                     [-2, +4, -2], \
                     [+1, -2, +1]])


def dxyyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the mixed fourth-order 
    derivative corresponding to a first-order derivative in the x-direction and 
    a third-order derivative in the y-direction.
    """
    return np.array([[-1/4, 0, +1/4], \
                     [+1/2, 0, -1/2], \
                     [   0, 0,    0], \
                     [-1/2, 0, +1/2], \
                     [+1/4, 0, -1/4]])


def dyyyymask() -> np.ndarray :
    """Defines a (minimal) mask for discrete approximation of the fourth-order 
    derivative in the y-direction.
    """
    return np.array([[+1], \
                     [-4], \
                     [+6], \
                     [-4], \
                     [+1]])


def computeNjetfcn(
        inpic,
        njetfcn : str,
        sigma : float,
        normdermethod : Union[str, ScSpNormDerMethod] = 'discgauss'
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
      'sqrtdetHessian' - signed square root of (absolute) determinant of the Hessian
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

    The derivatives in the (u, v)-coordinates are based on Equations (6)-(7) in
    Lindeberg (1998) "Edge detection and ridge detection with automatic scale 
    selection", International Journal of Computer Vision, vol 30(2): 117-154, 
    whereas the derivatives in the (p, q)-coordinates are based on 
    Equations (37)-(40) in the same article.

    This function is the preferred choice for simplicitly or if you only need a single 
    N-jetfcn at the given scale. If you instead want to compute multiple N-jetfcns
    at the same scale, it is, however, computationally more efficient to perform
    the scale-space smoothing yourself using the scspconv() function and then
    applying the function applyNjetfcn() multiple times for each N-jetfcn.
    """
    if isinstance(normdermethod, str):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    smoothpic = scspconv(inpic, sigma, normdermethod.scspmethod)

    return applyNjetfcn(smoothpic, njetfcn, sigma, normdermethod)


def applyNjetfcn(
        smoothpic : np.ndarray,
        njetfcn : str,
        sigma : float = 1.0,
        normdermethod : Union[str, ScSpNormDerMethod] = 'discgauss'
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
      'sqrtdetHessian' - signed square root of (absolute) determinant of the Hessian
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

    The derivatives in the (u, v)-coordinates are based on Equations (6)-(7) in
    Lindeberg (1998) "Edge detection and ridge detection with automatic scale 
    selection", International Journal of Computer Vision, vol 30(2): 117-154, 
    whereas the derivatives in the (p, q)-coordinates are based on 
    Equations (37)-(40) in the same article.
    """
    if isinstance(normdermethod, str):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    if ((smoothpic.ndim == 3) and (smoothpic.shape[2] > 1)):
        # Apply same function to all the layers if the input is a multi-layer image
        numlayers = smoothpic.shape[2]
        outpic = np.zeros(smoothpic.shape)
        for layer in range(0, numlayers):
            outpic[:, :, layer] = \
              applyNjetfcn(smoothpic[:, :, layer], njetfcn, sigma, normdermethod)

    else:
        if njetfcn == 'L':
            outpic = smoothpic

        elif njetfcn == 'Lx':
            outpic = normderfactor(1, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxmask())

        elif njetfcn == 'Ly':
            outpic = normderfactor(0, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dymask())

        elif njetfcn == 'Lxx':
            outpic = normderfactor(2, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxmask())

        elif njetfcn == 'Lxy':
            outpic = normderfactor(1, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxymask())

        elif njetfcn == 'Lyy':
            outpic = normderfactor(0, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dyymask())

        elif njetfcn == 'Lv':
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            outpic = np.sqrt(Lx*Lx + Ly*Ly)

        elif njetfcn == 'Lv2':
            Lx = normderfactor(1, 0, sigma, normdermethod) * \
                 correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * \
                 correlate(smoothpic, dymask())
            outpic = Lx*Lx + Ly*Ly

        elif njetfcn == 'Laplace':
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Lxx + Lyy

        elif njetfcn == 'detHessian':
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            outpic = Lxx*Lyy - Lxy*Lxy

        elif njetfcn == 'sqrtdetHessian':
            # Signed square root of absolute value of the determinant of the Hessian
            Lxx = normderfactor(2, 0, sigma, normdermethod) * \
                  correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * \
                  correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * \
                  correlate(smoothpic, dyymask())
            detHessian = Lxx*Lyy - Lxy*Lxy
            outpic = np.sign(detHessian) * np.sqrt(np.abs(detHessian))

        elif njetfcn == 'Kappa':
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

        elif njetfcn == 'Lv2Lvv':
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

        elif njetfcn == 'Lv3Lvvv':
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

        elif njetfcn == 'Lp':
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

        elif njetfcn == 'Lq':
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

        elif njetfcn == 'Lpp':
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

        elif njetfcn == 'Lqq':
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

        elif njetfcn == 'Lxxx':
            outpic = normderfactor(3, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxmask())

        elif njetfcn == 'Lxxy':
            outpic = normderfactor(2, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxxymask())

        elif njetfcn == 'Lxyy':
            outpic = normderfactor(1, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dxyymask())

        elif njetfcn == 'Lyyy':
            outpic = normderfactor(0, 3, sigma, normdermethod) * \
                     correlate(smoothpic, dyyymask())

        elif njetfcn == 'Lxxxx':
            outpic = normderfactor(4, 0, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxxmask())

        elif njetfcn == 'Lxxxy':
            outpic = normderfactor(3, 1, sigma, normdermethod) * \
                     correlate(smoothpic, dxxxymask())

        elif njetfcn == 'Lxxyy':
            outpic = normderfactor(2, 2, sigma, normdermethod) * \
                     correlate(smoothpic, dxxyymask())

        elif njetfcn == 'Lxyyy':
            outpic = normderfactor(1, 3, sigma, normdermethod) * \
                     correlate(smoothpic, dxyyymask())

        elif njetfcn == 'Lyyyy':
            outpic = normderfactor(0, 4, sigma, normdermethod) * \
                     correlate(smoothpic, dyyyymask())

        else:
            raise ValueError(f'NJetFcn {njetfcn} not implemented')

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
    if isinstance(normdermethod, str):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    if normdermethod.normdermethod == 'nonormalization':
        value = 1.0

    elif normdermethod.normdermethod == 'varnorm':
        # ==>> Here it could be natural to compensate for the additional variance
        # ==>> for the integrated or linearly integrated Gaussian kernels.
        # ==>> That added variance is, however, scale-dependent, which then
        # ==>> requires a separate calibration for each scale value
        value = sigma**(xorder + yorder)

    else:
        raise ValueError(f'Derivative method {normdermethod.normdermethod} \
not implemented')

    return value


def defaultscspnormdermethodobject(
        scspnormdermethod : str = 'discgaussvar',
        gamma : float = 1.0
    ) -> ScSpNormDerMethod :
    """Converts a user-friendly string for a method for approximating 
    scale-normalized derivatives for discrete data to a class object, including 
    also a specification of the discretization method to use for discrete 
    approximation of the underlying spatial smoothing operation
    """
    if scspnormdermethod == 'discgauss':
        obj = scspnormdermethodobject('discgauss', 'nonormalization', gamma)

    elif scspnormdermethod == 'discgaussvar':
        obj = scspnormdermethodobject('discgauss', 'varnorm', gamma)

    elif scspnormdermethod == 'samplgauss':
        obj = scspnormdermethodobject('samplgauss', 'nonormalization', gamma)

    elif scspnormdermethod == 'samplgaussvar':
        obj = scspnormdermethodobject('samplgauss', 'varnorm', gamma)

    elif scspnormdermethod == 'normsamplgauss':
        obj = scspnormdermethodobject('normsamplgauss', 'nonormalization', gamma)

    elif scspnormdermethod == 'normsamplgaussvar':
        obj = scspnormdermethodobject('normsamplgauss', 'varnorm', gamma)

    elif scspnormdermethod == 'intgauss':
        obj = scspnormdermethodobject('intgauss', 'nonormalization', gamma)

    elif scspnormdermethod == 'intgaussvar':
        obj = scspnormdermethodobject('intgauss', 'varnorm', gamma)

    elif scspnormdermethod == 'linintgauss':
        obj = scspnormdermethodobject('linintgauss', 'nonormalization', gamma)

    elif scspnormdermethod == 'linintgaussvar':
        obj = scspnormdermethodobject('linintgauss', 'varnorm', gamma)

    else:
        raise ValueError(f'Scale-space derivative method {scspnormdermethod} \
not implemented')

    return obj


def variance(spatfilter : np.ndarray) -> np.ndarray:
    """Returns the spatial covariance matrix of 2-D filter, assumed to be non-negative.
    """
    if spatfilter.ndim != 2:
        raise ValueError('Only implemented for 2-D filters so far')

    xsize = spatfilter.shape[1]
    ysize = spatfilter.shape[0]

    if xsize % 2:
        x = np.linspace(-(xsize-1)/2, (xsize-1)/2, xsize)
    else:
        # Choose convention to fit deltafcn()
        x = np.linspace(-xsize/2, xsize/2-1, xsize)

    if ysize % 2:
        y = np.linspace(-(ysize-1)/2, (ysize-1)/2, ysize)
    else:
        # Choose convention to fit deltafcn()
        y = np.linspace(-ysize/2, ysize/2-1, ysize)

    y = -y

    xv, yv = np.meshgrid(x, y, indexing='xy')

    x2mom = np.sum(np.sum(xv * xv * spatfilter))/np.sum(np.sum(spatfilter))
    xymom = np.sum(np.sum(xv * yv * spatfilter))/np.sum(np.sum(spatfilter))
    y2mom = np.sum(np.sum(yv * yv * spatfilter))/np.sum(np.sum(spatfilter))

    xmean, ymean = filtermean(spatfilter)

    return np.array([[x2mom - xmean*xmean, xymom - xmean*ymean], \
                     [xymom - xmean*ymean, y2mom - ymean*ymean]])


def filtermean(spatfilter : np.ndarray) -> (float, float) :
    """Returns the spatial mean vector of 2-D filter, assumed to be non-negative.
    """
    if spatfilter.ndim != 2:
        raise ValueError('Only implemented for 2-D filters so far')

    xsize = spatfilter.shape[1]
    ysize = spatfilter.shape[0]

    if xsize % 2:
        x = np.linspace(-(xsize-1)/2, (xsize-1)/2, xsize)
    else:
        # Choose convention to fit deltafcn()
        x = np.linspace(-xsize/2, xsize/2-1, xsize)

    if ysize % 2:
        y = np.linspace(-(ysize-1)/2, (ysize-1)/2, ysize)
    else:
        # Choose convention to fit deltafcn()
        y = np.linspace(-ysize/2, ysize/2-1, ysize)

    y = -y

    xv, yv = np.meshgrid(x, y, indexing='xy')

    xmean = np.sum(np.sum(xv * spatfilter))/np.sum(np.sum(spatfilter))
    ymean = np.sum(np.sum(yv * spatfilter))/np.sum(np.sum(spatfilter))

    return xmean, ymean


def mean1D(spatfilter : np.ndarray) -> float:
    """Computes the spatial mean of a non-negative 1-D filter.
    """
    if spatfilter.ndim != 1:
        raise ValueError('Only implemented for 1-D filters')

    size = spatfilter.shape[0]
    x = np.linspace(0, size-1, size)

    return np.sum(np.sum(x * spatfilter)) / np.sum(np.sum(spatfilter))


def variance1D(spatfilter : np.ndarray) -> float:
    """Computes the spatial variance of a non-negative 1-D filter.
    """
    if spatfilter.ndim != 1:
        raise ValueError('Only implemented for 1-D filters')

    size = spatfilter.shape[0]
    x = np.linspace(0, size-1, size)

    x2mom = np.sum(np.sum(x * x * spatfilter)) / np.sum(np.sum(spatfilter))
    xmean = mean1D(spatfilter)

    return x2mom - xmean * xmean


def RGB2LUV(inpic) -> np.ndarray:
    """Converts an RGB colour image to a colour-opponent LUV colour space.
    """
    inpic = np.array(inpic).astype('float')

    outpic = np.zeros(inpic.shape)
    outpic[:, :, 0] = (inpic[:, :, 0] + inpic[:, :, 1] + inpic[:, :, 2])/3.0
    outpic[:, :, 1] = 1.0*(inpic[:, :, 0] - inpic[:, :, 1])
    outpic[:, :, 2] = (inpic[:, :, 0] + inpic[:, :, 1])/2.0 - inpic[:, :, 2]

    return outpic


def RGB2L(inpic) -> np.ndarray:
    """Converts an RGB colour image to a greylevel image.
    """
    inpic = np.array(inpic).astype('float')

    return (inpic[:, :, 0] + inpic[:, :, 1] + inpic[:, :, 2])/3.0


def applydirder(
        smoothpic : np.ndarray,
        phi : float,
        phiorder : int,
        orthorder : int = 0,
        sigma : float = 1.0,
        normdermethod : str = 'varnorm'
    ) -> np.ndarray :
    """Applies a directional derivative, of order phiorder in the direction phi
    and of order orthorder in the orthogonal direction, to the input image.

    The directional operator is defined as

    D_phi^phiorder D_orth^orthorder

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_x and D_y constitute (discrete approximations of) partial derivative
    operators along the x- and y-directions, respectively.

    The intention is that if this operation is applied to image data, then
    it should be preceeded by a call to the spatial smoothing operation,
    as can be performed by e.g. the function scspconv(). If you want to
    make use of scale-normalized derivatives, the parameter sigma should
    then describe the value of the spatial scale parameter used for spatial
    smoothing in units of the standard deviation of the kernel.

    References:

    Lindeberg (1993) "Scale-Space Theory in Computer Vision", Springer.
    (See Equation (5.54) on page 139.)

    Lindeberg (2013) "A computational theory of visual receptive fields", 
    Biological Cybernetics, 107(6): 589-635. (See Equation (69).)

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (31) and the explanation
    to Equation (23).)
    """
    # Determine a directional derivative approximation mask
    mask = dirdermask(phi, phiorder, orthorder)

    # Determine the scale normalization factor
    if normdermethod == 'varnorm':
        scalenormfactor = sigma**(phiorder + orthorder)
    elif normdermethod == 'nonormalization':
        scalenormfactor = 1.0
    else:
        raise ValueError(f'Scale normalization method {normdermethod} not implemented')

    return scalenormfactor * correlate(smoothpic, mask)


def dirdermask(
        phi : float,
        phiorder : int,
        orthorder : int = 0,
    ) -> np.ndarray :
    """Returns a directional derivative mask of order phiorder in the direction phi
    and order orthorder in the orthogonal direction.

    The mask will be of size 3 x 3 if the total order of differentiation is either
    1 or 2, whereas the mask will be of size 5 x 5 if the total order of differentiation
    is either 3 or 4. If the total order of differentiation is 0, then the mask will 
    be of size 1 x 1.
    """
    if (phiorder == 1) and (orthorder == 0):
        return dphi_mask(phi)
    if (phiorder == 2) and (orthorder == 0):
        return dphiphi_mask(phi)
    if (phiorder == 0) and (orthorder == 1):
        return dorth_mask(phi)
    if (phiorder == 0) and (orthorder == 2):
        return dorthorth_mask(phi)
    if (phiorder == 1) and (orthorder == 1):
        return dphiorth_mask(phi)
    if (phiorder == 3) and (orthorder == 0):
        return dphiphiphi_mask(phi)
    if (phiorder == 2) and (orthorder == 1):
        return dphiphiorth_mask(phi)
    if (phiorder == 1) and (orthorder == 2):
        return dphiorthorth_mask(phi)
    if (phiorder == 0) and (orthorder == 3):
        return dorthorthorth_mask(phi)
    if (phiorder == 4) and (orthorder == 0):
        return dphiphiphiphi_mask(phi)
    if (phiorder == 3) and (orthorder == 1):
        return dphiphiphiorth_mask(phi)
    if (phiorder == 2) and (orthorder == 2):
        return dphiphiorthorth_mask(phi)
    if (phiorder == 1) and (orthorder == 3):
        return dphiorthorthorth_mask(phi)
    if (phiorder == 0) and (orthorder == 4):
        return dorthorthorthorth_mask(phi)
    if (phiorder == 0) and (orthorder == 0):
        return np.array([[1]])

    raise ValueError(f"Not implemented directional derivatives of order \
                     (phiorder {phiorder}) and (orthorder {orthorder})")


def dphi_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the first-order directional 
    derivative in the orientation phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer.
    """
    return cos(phi) * dxmask3() + sin(phi) * dymask3()


def dphiphi_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the second-order directional 
    derivative in the orientation phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer.
    """
    return cos(phi)**2 * dxxmask3() + \
           2*cos(phi)*sin(phi) * dxymask3() + \
           sin(phi)**2 * dyymask3()


def dorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the first-order directional 
    derivative in a perpendicular orientation to phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return -sin(phi) * dxmask3() + cos(phi) * dymask3()


def dorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the second-order directional 
    derivative in a perpendicular orientation to phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return sin(phi)**2 * dxxmask3() \
           - 2*cos(phi)*sin(phi) * dxymask3() \
           + cos(phi)**2 * dyymask3()


def dphiorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed second-order directional 
    derivative in the directions of phi and its perpendicular orientation.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return cos(phi)*sin(phi) * (dyymask3() - dxxmask3()) \
           + (cos(phi)**2 - sin(phi)**2) * dxymask3()


def dxmask3() -> np.ndarray :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[  0,  0,   0 ], \
                     [-1/2, 0, +1/2], \
                     [  0,  0,   0 ]])


def dymask3() -> np.ndarray :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    first-order derivative in the y-direction.
    """
    return np.array([[0, +1/2, 0], \
                     [0,   0,  0], \
                     [0, -1/2, 0]])


def dxxmask3() -> np.ndarray :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    second-order derivative in the x-direction.
    """
    return np.array([[0,  0,  0], \
                     [1, -2,  1], \
                     [0,  0,  0]])


def dxymask3() -> np.ndarray :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    mixed second-order derivative in the x- and y-directions.
    """
    return np.array([[-1/4, 0, +1/4], \
                     [   0, 0,    0], \
                     [+1/4, 0, -1/4]])


def dyymask3() -> np.ndarray :
    """Defines a mask of size 3 x 3 for discrete approximation of the 
    second-order derivative in the y-direction.
    """
    return np.array([[0, +1, 0], \
                     [0, -2, 0], \
                     [0, +1, 0]])


def dphiphiphi_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the third-order directional 
    derivative in the orientation phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer.
    """
    return cos(phi)**3 * dxxxmask5() + \
           3*cos(phi)**2 * sin(phi) * dxxymask5() + \
           3*cos(phi) * sin(phi)**2 * dxyymask5() + \
           sin(phi)**3 * dyyymask5()


def dphiphiorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed third-order directional 
    derivative, corresponding to a second-order directional derivative in the 
    orientation phi and a first-order directional derivative in the orthogonal 
    direction.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return - cos(phi)**2 * sin(phi) * dxxxmask5() \
           + (cos(phi)**3 - 2 * cos(phi) * sin(phi)**2) * dxxymask5() \
           - (sin(phi)**3 - 2 * cos(phi)**2 * sin(phi)) * dxyymask5() \
           + cos(phi) * sin(phi)**2 * dyyymask5()


def dphiorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed third-order directional 
    derivative, corresponding to a first-order directional derivative in the 
    orientation phi and a second-order directional derivative in the orthogonal 
    direction.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return cos(phi) * sin(phi)**2 * dxxxmask5() \
           + (sin(phi)**3 - 2 * cos(phi)**2 * sin(phi)) * dxxymask5() \
           + (cos(phi)**3 - 2 * cos(phi) * sin(phi)**2) * dxyymask5() \
           + cos(phi)**2 * sin(phi) * dyyymask5()


def dorthorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the third-order directional 
    derivative in the orientation orthogonal to phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return - sin(phi)**3 * dxxxmask5() \
           + 3*sin(phi)**2 * cos(phi) * dxxymask5() \
           - 3*sin(phi) * cos(phi)**2 * dxyymask5() \
           + cos(phi)**3 * dyyymask5()


def dphiphiphiphi_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the fourth-order directional 
    derivative in the orientation phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer.
    """
    return cos(phi)**4 * dxxxxmask5() \
           + 4 * cos(phi)**3 * sin(phi) * dxxxymask5() \
           + 6 * cos(phi)**2 * sin(phi)**2 * dxxyymask5() \
           + 4 * cos(phi) * sin(phi)**3 * dxyyymask5() \
           + sin(phi)**4 * dyyyymask5()


def dphiphiphiorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed fourth-order directional 
    derivative corresponding to a third-order directional derivative in the orientation 
    phi and a first-order directional derivative in the orthogonal direction.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return - cos(phi)**3 * sin(phi) * dxxxxmask5() \
           + (cos(phi)**4 - 3 * cos(phi)**2 * sin(phi)**2) * dxxxymask5() \
           + 3 * (cos(phi)**3 * sin(phi) - cos(phi) * sin(phi)**3) * dxxyymask5() \
           + (3 * cos(phi)**2 * sin(phi)**2 - sin(phi)**4) * dxyyymask5() \
           + cos(phi) * sin(phi)**3 * dyyyymask5()


def dphiphiorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed fourth-order directional 
    derivative corresponding to a second-order directional derivative in the 
    orientation phi and a second-order directional derivative in the orthogonal 
    direction.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return cos(phi)**2 * sin(phi)**2 * dxxxxmask5() \
           + 2 * (cos(phi) * sin(phi)**3 - cos(phi)**3 * sin(phi)) * dxxxymask5() \
           + (cos(phi)**4 - 4 * cos(phi)**2 * sin(phi)**2 + sin(phi)**4) \
             * dxxyymask5() \
           + 2 * (cos(phi)**3 * sin(phi) - cos(phi) * sin(phi)**3) * dxyyymask5() \
           + cos(phi)**2 * sin(phi)**2 * dyyyymask5()


def dphiorthorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the mixed fourth-order directional 
    derivative corresponding to a first-order directional derivative in the orientation 
    phi and a third-order directional derivative in the orthogonal direction.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return - cos(phi) * sin(phi)**3 * dxxxxmask5() \
           + (3 * cos(phi)**2 * sin(phi)**2 - sin(phi)**4) * dxxxymask5() \
           + 3 * (cos(phi) * sin(phi)**3 - cos(phi)**3 * sin(phi)) * dxxyymask5() \
           + (cos(phi)**4 - 3 * cos(phi)**2 * sin(phi)**2) * dxyyymask5() \
           + cos(phi)**3 * sin(phi) * dyyyymask5()


def dorthorthorthorth_mask(phi : float) -> np.ndarray :
    """Defines a mask for discrete approximation of the fourth-order directional 
    derivative in the orientation orthogonal to phi.

    See Equation (5.54) on page 139 in Lindeberg (1993) 
    "Scale-Space Theory in Computer Vision", Springer,
    and note that an orthogonal direction to the unit vector 
    (cos phi, sin phi) is (-sin phi, cos phi).
    """
    return sin(phi)**4 * dxxxxmask5() \
           - 4 * sin(phi)**3 * cos(phi) * dxxxymask5() \
           + 6 * sin(phi)**2 * cos(phi)**2 * dxxyymask5() \
           - 4 * sin(phi) * cos(phi)**3 * dxyyymask5() \
           + cos(phi)**4 * dyyyymask5()


def dxmask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    first-order derivative in the x-direction.
    """
    return np.array([[0,   0,  0,   0,  0], \
                     [0,   0,  0,   0,  0], \
                     [0, -1/2, 0, +1/2, 0], \
                     [0,   0,  0,   0,  0], \
                     [0,   0,  0,   0,  0]])


def dymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    first-order derivative in the y-direction.
    """
    return np.array([[0, 0,   0,  0, 0], \
                     [0, 0, +1/2, 0, 0], \
                     [0, 0,   0,  0, 0], \
                     [0, 0, -1/2, 0, 0], \
                     [0, 0,   0,  0, 0]])


def dxxmask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    second-order derivative in the x-direction.
    """
    return np.array([[0, 0,  0,  0, 0], \
                     [0, 0,  0,  0, 0], \
                     [0, 1, -2,  1, 0], \
                     [0, 0,  0,  0, 0], \
                     [0, 0,  0,  0, 0]])


def dxymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    mixed second-order derivative in the x- and y-directions.
    """
    return np.array([[0,    0, 0,    0, 0], \
                     [0, -1/4, 0, +1/4, 0], \
                     [0,    0, 0,    0, 0], \
                     [0, +1/4, 0, -1/4, 0], \
                     [0,    0, 0,    0, 0]])


def dyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    second-order derivative in the y-direction.
    """
    return np.array([[0, 0,  0, 0, 0], \
                     [0, 0, +1, 0, 0], \
                     [0, 0, -2, 0, 0], \
                     [0, 0, +1, 0, 0], \
                     [0, 0,  0, 0, 0]])


def dxxxmask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the 
    third-order derivative in the x-direction.
    """
    return np.array([[  0,   0, 0,  0,  0 ], \
                     [  0,   0, 0,  0,  0 ], \
                     [-1/2, +1, 0, -1, 1/2], \
                     [  0,   0, 0,  0,  0 ], \
                     [  0,   0, 0,  0,  0 ]])


def dxxymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the mixed 
    third-order derivative corresponding to a second-order derivative in the 
    x-direction and a first-order derivative in the y-direction.
    """
    return np.array([[0,   0,   0,   0,  0], \
                     [0, +1/2, -1, +1/2, 0], \
                     [0,   0,   0,   0,  0], \
                     [0, -1/2, +1, -1/2, 0], \
                     [0,   0,   0,   0,  0]])


def dxyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the mixed 
    third-order derivative corresponding to a first-order derivative in the 
    x-direction and a second-order derivative in the y-direction.
    """
    return np.array([[0,   0,  0,   0,  0], \
                     [0, -1/2, 0, +1/2, 0], \
                     [0,   +1, 0,   -1, 0], \
                     [0, -1/2, 0, +1/2, 0], \
                     [0,   0,  0,   0,  0]])


def dyyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the third-order 
    derivative in the y-direction.
    """
    return np.array([[0, 0, +1/2, 0, 0], \
                     [0, 0,   -1, 0, 0], \
                     [0, 0,    0, 0, 0], \
                     [0, 0,   +1, 0, 0], \
                     [0, 0, -1/2, 0, 0]])


def dxxxxmask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the fourth-order 
    derivative in the x-direction.
    """
    return np.array([[0,  0, 0,  0, 0], \
                     [0,  0, 0,  0, 0], \
                     [1, -4, 6, -4, 1], \
                     [0,  0, 0,  0, 0], \
                     [0,  0, 0,  0, 0]])


def dxxxymask5() -> np.ndarray :
    """Defines a of size 5 x 5 mask for discrete approximation of the mixed 
    fourth-order derivative corresponding to a third-order derivative in the 
    x-direction and a first-order derivative in the y-direction.
    """
    return np.array([[   0,    0, 0,    0,    0], \
                     [-1/4, +1/2, 0, -1/2, +1/4], \
                     [   0,    0, 0,    0,    0], \
                     [+1/4, -1/2, 0, +1/2, -1/4], \
                     [   0,    0, 0,    0,    0]])


def dxxyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the mixed 
    fourth-order derivative corresponding to a second-order derivatives in 
    the x- and y-directions.
    """
    return np.array([[0,  0,  0,  0, 0], \
                     [0, +1, -2, +1, 0], \
                     [0, -2, +4, -2, 0], \
                     [0, +1, -2, +1, 0], \
                     [0,  0,  0,  0, 0]])


def dxyyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the mixed 
    fourth-order derivative corresponding to a first-order derivative in the 
    x-direction and a third-order derivative in the y-direction.
    """
    return np.array([[0, -1/4, 0, +1/4, 0], \
                     [0, +1/2, 0, -1/2, 0], \
                     [0,    0, 0,    0, 0], \
                     [0, -1/2, 0, +1/2, 0], \
                     [0, +1/4, 0, -1/4, 0]])


def dyyyymask5() -> np.ndarray :
    """Defines a mask of size 5 x 5 for discrete approximation of the fourth-order 
    derivative in the y-direction.
    """
    return np.array([[0, 0, +1, 0, 0], \
                     [0, 0, -4, 0, 0], \
                     [0, 0, +6, 0, 0], \
                     [0, 0, -4, 0, 0], \
                     [0, 0, +1, 0, 0]])
