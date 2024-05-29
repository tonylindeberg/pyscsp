"""Gaussian Derivative Toolbox for Python

For computing convolutions with Gaussian derivative kernels, using discretizations
in terms of either sampled Gaussian derivative kernels or integrated Gaussian 
derivative kernels. Then, different types of feature detectors can be defined, 
by combining responses to different Gaussian derivative operators into differential 
expressions.

Note: The scale normalization does not explicitly compensate for det additional
variance 1/12 for the integrated Gaussian kernel at coarser scales.

References:

Koenderink and van Doorn (1992) "Generic neighborhood operators", IEEE Transactions
on Pattern Analysis and Machine Intelligence, 14(6): 597-605.

Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.

Lindeberg (1998a) "Feature detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 77-116.

Lindeberg (1998b) "Edge detection and ridge detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 117-154.

Lindeberg (2015) "Image matching using generalized scale-space interest points", 
Journal of Mathematical Imaging and Vision, 52(1): 3-36.

Limitations:

Compared to the full theory of scale-normalized Gaussian derivative operators with
their associated differential invariants, the following implementation is reduced 
in the following ways:

- much fewer functions of the N-jet have so far been implemented,
- there is no passing of additional parameters to functions of the N-jet.

Furthermore, this implementation has not been thoroughly tested.
"""


from math import sqrt, pi, ceil, exp, erf
from typing import NamedTuple, Union
import numpy as np
from scipy.ndimage import correlate1d

from pyscsp.discscsp import scaled_erf, truncerrtransf, \
                            gaussfiltsize, make1Ddiscgaussderfilter, \
                            make1Dsamplgaussdifffilter, \
                            make1Dnormsamplgaussdifffilter, \
                            make1Dintgaussdifffilter


class GaussDerMethod(NamedTuple):
    """Object for storing the parameters of a Gaussian derivative discretization  
    and scale normalization method.
    """
    methodname: str      # either 'samplgaussder' or 'intgaussder'
    normdermethod: str   # either 'varnorm' or 'nonormalization'


def gaussderconv(
        inpic,
        xorder : int,
        yorder : int,
        sigma : float,
        gaussdernormmethod : Union[str, GaussDerMethod] = 'samplgaussder',
        gamma : float = 1.0,
        epsilon : float = 0.0001
        ) -> np.ndarray :
    """Convolves the 2-D image inpic with a sampled Gaussian derivative kernel of order
    xsize in the x-direction and of order ysize in the y-direction, with standard
    deviation sigma.

    The parameter gaussdernormmethod should provide a combined specification of method
    to be used for discretizing the Gaussian derivative convolution operation as well
    as the method to be used for scale normalization. The following methods are 
    provided:

    'samplgaussder'        - the sampled Gaussian derivative kernel without scale 
                             normalization
    'samplgaussdervar'     - the sampled Gaussian derivative kernel with variance-based 
                             scale normalization
    'intgaussder'          - the integrated Gaussian derivative kernel without 
                             scale normalization
    'intgaussdervar'       - the integrated Gaussian derivative kernel with 
                             variance-based scale normalization

    The different discretization methods have the following relative advantages (+)
    and disadvantages (-):

    'samplgaussder': + no added scale offset in the spatial discretization
                     - for small values of sigma, the discrete kernel values may 
                       sum up to a value larger than the integral of the 
                       corresponding continuous kernel
                     - for very small values of sigma, the kernels have a too 
                       narrow shape

    'intgaussder': + the discrete kernel values may sum up to a value closer
                     to the L1-norm of the corresponding continuous kernel over 
                     an infinite domain
                   - the box integration introduces a scale offset of 1/12 at 
                     coarser scales

    The parameter epsilon should specify an upper bound on the relative truncation
    error, while the parameter gamma should denote the scale normalization power
    in the scale-normalized derivative concept.
    """
    if isinstance(gaussdernormmethod, str):
        gaussdernormmethod = defaultgaussdernormdermethodobject(gaussdernormmethod)

    methodname = gaussdernormmethod.methodname
    normdermethod = gaussdernormmethod.normdermethod

    # Estimate a size for truncating the Gaussian derivative kernels
    N = N_from_epsilon_2D(max(xorder, yorder), sigma, epsilon)

    if methodname == 'samplgaussder':
        return samplgaussderconv(inpic, xorder, yorder, sigma, N, \
                                 normdermethod, gamma)

    if methodname == 'intgaussder':
        return intgaussderconv(inpic, xorder, yorder, sigma, N, \
                               normdermethod, gamma)

    raise ValueError(f"Gaussian derivative discretization method {methodname} \
not implemented")


def samplgaussderconv(
        inpic,
        xorder : int,
        yorder : int,
        sigma : float,
        N : int,
        normdermethod : str = 'nonormalization',
        gamma : float = 1.0
        ) -> np.ndarray :
    """Convolves the 2-D image inpic with a sampled Gaussian derivative kernel of order
    xorder in the x-direction and of order yorder in the y-direction, with standard
    deviation sigma, and truncated at +/- N along the two coordinate axes.

    The parameter normdermethod specifies the scale normalization method 
    (either 'varnorm' or 'nonormalization') with gamma denoting the scale normalization
    power in the scale-normalized derivative concept.
    """
    xfilter = make1Dsamplgaussderfilter(xorder, sigma, N)
    yfilter = make1Dsamplgaussderfilter(yorder, sigma, N)

    if normdermethod == 'varnorm':
        scalenormfactor = sigma**((xorder + yorder) * gamma)

    elif normdermethod == 'nonormalization':
        scalenormfactor = 1.0

    else:
        raise ValueError(f"Scale normalization method {normdermethod} not implemented")

    # For the separable convolution operation, we only need to flip the filter in the
    # x-direction, since our convention for the y-direction is positive upwards, and
    # therefore opposite in relation to the correlation functions that we build upon.
    tmppic = correlate1d(np.array(inpic).astype('float'), yfilter, 0)
    outpic = correlate1d(tmppic, np.flip(xfilter), 1)

    return scalenormfactor * outpic


def make1Dsamplgaussderfilter(
        order : int,
        sigma : float,
        N : int
        ) -> np.ndarray :
    """Generates a sampled Gaussian derivative kernel of a given order and with 
    standard deviation sigma, truncated at the ends at -N and -N.
    """
    x = np.linspace(-N, N, 1 + 2*N)

    if order == 0:
        return gauss0der(x, sigma)

    if order == 1:
        return gauss1der(x, sigma)

    if order == 2:
        return gauss2der(x, sigma)

    if order == 3:
        return gauss3der(x, sigma)

    if order == 4:
        return gauss4der(x, sigma)

    raise ValueError(f"Not implemented for order {order}")


def gauss0der(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes a Gaussian function, given a set of spatial x-coordinates and 
    sigma value specifying the standard deviation of the kernel.
    """
    return 1 / (sqrt( 2 * pi) * sigma) * np.exp(-(x**2 / (2 * sigma**2)))


def gauss1der(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes a first-order derivative of a Gaussian function given a set of spatial 
    x-coordinates and sigma value specifying the standard deviation of the kernel.
    """
    return (-x / sigma**2) / (sqrt(2 * pi) * sigma) * np.exp(-(x**2 / (2 * sigma**2)))


def gauss2der(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes a second-order derivative of a Gaussian function given a set of spatial 
    x-coordinates and sigma value specifying the standard deviation of the kernel.
    """
    return ((x**2 - sigma**2) / sigma**4) / \
           (sqrt(2 * pi) * sigma) * np.exp(-(x**2 / (2 * sigma**2)))


def gauss3der(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes a third-order derivative of a Gaussian function given a set of spatial 
    x-coordinates and sigma value specifying the standard deviation of the kernel.
    """
    return (-(x**3 - 3 * sigma**2 * x) / sigma**6) / \
           (sqrt(2 * pi) * sigma) * np.exp(-(x**2 / (2 * sigma**2)))


def gauss4der(x : np.ndarray, sigma : float = 1.0) -> np.ndarray :
    """Computes a fourth-order derivative of a Gaussian function given a set of spatial 
    x-coordinates and sigma value specifying the standard deviation of the kernel.
    """
    return ((x**4 - 6 * sigma**2 * x**2 + 3 * sigma**4) / sigma**8) / \
           (sqrt(2 * pi) * sigma) * np.exp(-(x**2 / (2 * sigma**2)))


def L1norm(kernel : np.ndarray) -> float :
    """Returns the L1-norm of a filter kernel"""
    return np.sum(np.abs(kernel))


def intgaussderconv(
        inpic,
        xorder : int,
        yorder : int,
        sigma : float,
        N : int,
        normdermethod : str = 'nonormalization',
        gamma : float = 1.0
        ) -> np.ndarray :
    """Convolves the 2-D image inpic with an integrated sampled Gaussian derivative 
    kernel of order xorder in the x-direction and of order yorder in the y-direction, 
    with standard deviation sigma, and truncated at +/- N along the two coordinate axes.

    The integrated Gaussian derivative kernel is defined by integrating the 
    corresponding continuous Gaussian derivative kernel over the support region 
    of each pixel.

    The parameter normdermethod specifies the scale normalization method 
    (either 'varnorm' or 'nonormalization') with gamma denoting the scale normalization
    power in the scale-normalized derivative concept.
    """
    xfilter = make1Dintgaussderfilter(xorder, sigma, N)
    yfilter = make1Dintgaussderfilter(yorder, sigma, N)

    if normdermethod == 'varnorm':
        scalenormfactor = sigma**((xorder + yorder) * gamma)

    elif normdermethod == 'nonormalization':
        scalenormfactor = 1.0

    else:
        raise ValueError(f"Scale normalization method {normdermethod} not implemented")

    # For the separable convolution operation, we only need to flip the filter in the
    # x-direction, since our convention for the y-direction is positive upwards, and
    # therefore opposite in relation to the correlation functions that we build upon.
    tmppic = correlate1d(np.array(inpic).astype('float'), yfilter, 0)
    outpic = correlate1d(tmppic, np.flip(xfilter), 1)

    return scalenormfactor * outpic


def make1Dintgaussderfilter(
        order : int,
        sigma : float,
        N : int
        ) -> np.ndarray :
    """Generates an integrated Gaussian derivative kernel of a given order and with 
    standard deviation sigma, truncated at the ends at -N and -N.

    The integrated Gaussian derivative kernel is defined by integrating the 
    corresponding continuous Gaussian derivative kernel over the support region 
    of each pixel.

    Note: At coarser scales, the box integration over each pixel support
    regions adds a scale offset to the kernel.
    """
    x = np.linspace(-N, N, 1 + 2*N)

    if order == 0:
        return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)

    if order == 1:
        return gauss0der(x + 0.5, sigma) - gauss0der(x - 0.5, sigma)

    if order == 2:
        return gauss1der(x + 0.5, sigma) - gauss1der(x - 0.5, sigma)

    if order == 3:
        return gauss2der(x + 0.5, sigma) - gauss2der(x - 0.5, sigma)

    if order == 4:
        return gauss3der(x + 0.5, sigma) - gauss3der(x - 0.5, sigma)

    raise ValueError(f"Not implemented for order {order}")


def N_from_epsilon_2D(
        order : int,
        sigma : float,
        epsilon : float
    ) -> int :
    """Computes an estimate of a minimum bound N for truncating a Gaussian
    derivative kernel of a given order to ensure that the relative truncation 
    error for 2-D Gaussian derivative convolution is below epsilon.
    """
    # Convert the 2-D error bound for a 1-D error bound for separable convolution
    eps1D = truncerrtransf(epsilon, 2)

    return N_from_epsilon_1D(order, sigma, eps1D)


def N_from_epsilon_1D(
        order : int,
        sigma : float,
        epsilon : float
    ) -> int :
    """Computes an estimate of a minimum bound N for truncating a Gaussian
    derivative kernel of a given order to ensure that the relative truncation 
    error for 1-D Gaussian derivative convolution is below epsilon.
    """
    if order == 0:
        return ceil(gaussfiltsize(sigma, epsilon, 1))

    # ==>> Preliminary solution to be similar to corresponding error estimates
    # ==>> for the discrete approximations to Gaussian derivative responsens
    # ==>> computed by applying small-support discrete derivative approximation
    # ==>> molecules to the result of discrete scale-space smoothing
    return 1 + order + ceil(gaussfiltsize(sigma, epsilon / 2**order, 1))


def defaultgaussdernormdermethodobject(
        gaussdernormmethod : str
    ) -> GaussDerMethod :
    """Returns an object that specifies the parameters of a Gaussian derivative 
    discretization and scale normalization method.
    """
    if gaussdernormmethod == 'samplgaussdervar':
        return GaussDerMethod('samplgaussder', 'varnorm')

    if gaussdernormmethod == 'samplgaussder':
        return GaussDerMethod('samplgaussder', 'nonormalization')

    if gaussdernormmethod == 'intgaussdervar':
        return GaussDerMethod('intgaussder', 'varnorm')

    if gaussdernormmethod == 'intgaussder':
        return GaussDerMethod('intgaussder', 'nonormalization')

    raise ValueError(f"Unknown Gaussian derivative operator discretization method \
{gaussdernormmethod}")


def gaussderNjetfcn(
        inpic,
        njetfcn : str,
        sigma : float,
        gaussdernormmethod : Union[str, GaussDerMethod] = 'samplgaussder',
        gamma : float = 1.0,
        epsilon : float = 0.0001
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

    The parameter gaussdernormmethod should provide a combined specification of method
    to be used for discretizing the Gaussian derivative convolution operation as well
    as the method to be used for scale normalization. The following methods are 
    provided:

    'samplgaussder'        - the sampled Gaussian derivative kernel without scale 
                             normalization
    'samplgaussdervar'     - the sampled Gaussian derivative kernel with variance-based 
                             scale normalization
    'intgaussder'          - the integrated Gaussian derivative kernel without 
                             scale normalization
    'intgaussdervar'       - the integrated Gaussian derivative kernel with 
                             variance-based scale normalization

    The different discretization methods have the following relative advantages (+)
    and disadvantages (-):

    'samplgaussder': + no added scale offset in the spatial discretization
                     - for small values of sigma, the discrete kernel values may 
                       sum up to a value larger than the integral of the 
                       corresponding continuous kernel
                     - for very small values of sigma, the kernels have a too 
                       narrow shape

    'intgaussder': + the discrete kernel values may sum up to a value closer
                     to the L1-norm of the corresponding continuous kernel over 
                     an infinite domain
                   - the box integration introduces a scale offset of 1/12 at 
                     coarser scales

    The parameter epsilon should specify an upper bound on the relative truncation
    error, while the parameter gamma should denote the scale normalization power
    in the scale-normalized derivative concept.
    """
    if isinstance(gaussdernormmethod, str):
        gaussdernormmethod = defaultgaussdernormdermethodobject(gaussdernormmethod)

    if ((inpic.ndim == 3) and (inpic.shape[2] > 1)):
        # Apply same function to all the layers if the input is a multi-layer image
        numlayers = inpic.shape[2]
        outpic = np.zeros(inpic.shape)
        for layer in range(0, numlayers):
            outpic[:, :, layer] = \
              gaussderNjetfcn(inpic[:, :, layer], njetfcn, sigma, \
                              gaussdernormmethod, gamma, epsilon)

    else:
        if njetfcn == 'L':
            outpic = gaussderconv(inpic, 0, 0, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lx':
            outpic = gaussderconv(inpic, 1, 0, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Ly':
            outpic = gaussderconv(inpic, 0, 1, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxx':
            outpic = gaussderconv(inpic, 2, 0, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxy':
            outpic = gaussderconv(inpic, 1, 1, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lyy':
            outpic = gaussderconv(inpic, 0, 2, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lv':
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            outpic = np.sqrt(Lx*Lx + Ly*Ly)

        elif njetfcn == 'Lv2':
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            outpic = Lx*Lx + Ly*Ly

        elif njetfcn == 'Laplace':
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = Lxx + Lyy

        elif njetfcn == 'detHessian':
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = Lxx*Lyy - Lxy*Lxy

        elif njetfcn == 'sqrtdetHessian':
            # Signed square root of absolute value of the determinant of the Hessian
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            detHessian = Lxx*Lyy - Lxy*Lxy
            outpic = np.sign(detHessian) * np.sqrt(np.abs(detHessian))

        elif njetfcn == 'Kappa':
            # Rescaled level curve curvature
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = Ly*Ly*Lxx + Lx*Lx*Lyy - 2*Lx*Ly*Lxy

        elif njetfcn == 'Lv2Lvv':
            # 2nd-order derivative in gradient direction (used for edge detection)
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = Lx*Lx*Lxx + 2*Lx*Ly*Lxy + Ly*Ly*Lyy

        elif njetfcn == 'Lv3Lvvv':
            # 3rd-order derivative in gradient direction (used for edge detection)
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxxx = gaussderconv(inpic, 3, 0, sigma, \
                                gaussdernormmethod, gamma, epsilon)
            Lxxy = gaussderconv(inpic, 2, 1, sigma, \
                                gaussdernormmethod, gamma, epsilon)
            Lxyy = gaussderconv(inpic, 1, 2, sigma, \
                                gaussdernormmethod, gamma, epsilon)
            Lyyy = gaussderconv(inpic, 0, 3, sigma, \
                                gaussdernormmethod, gamma, epsilon)
            outpic = Lx*Lx*Lx*Lxxx + 3*Lx*Lx*Ly*Lxxy + \
                     3*Lx*Ly*Ly*Lxyy + Ly*Ly*Ly*Lyyy

        elif njetfcn == 'Lp':
            # 1st-order derivative in first principal curvature direction
            # (used for ridge detection)
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + \
                                np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = sinbeta * Lx - cosbeta * Ly

        elif njetfcn == 'Lq':
            # 1st-order derivative in second principal curvature
            # (used for valley detection)
            Lx = gaussderconv(inpic, 1, 0, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Ly = gaussderconv(inpic, 0, 1, sigma, \
                              gaussdernormmethod, gamma, epsilon)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + \
                                np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + \
                                np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = cosbeta * Lx + sinbeta * Ly

        elif njetfcn == 'Lpp':
            # 2nd-order derivative in first principal curvature direction
            # (used for ridge detection)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = ((Lxx + Lyy) - \
                      np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2

        elif njetfcn == 'Lqq':
            # 2nd-order derivative in second principal curvature direction
            # (used for valley detection)
            Lxx = gaussderconv(inpic, 2, 0, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lxy = gaussderconv(inpic, 1, 1, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            Lyy = gaussderconv(inpic, 0, 2, sigma, \
                               gaussdernormmethod, gamma, epsilon)
            outpic = ((Lxx + Lyy) + \
                      np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2

        elif njetfcn == 'Lxxx':
            outpic = gaussderconv(inpic, 3, 0, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxxy':
            outpic = gaussderconv(inpic, 2, 1, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxyy':
            outpic = gaussderconv(inpic, 1, 2, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lyyy':
            outpic = gaussderconv(inpic, 0, 3, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxxxx':
            outpic = gaussderconv(inpic, 4, 0, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxxxy':
            outpic = gaussderconv(inpic, 3, 1, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxxyy':
            outpic = gaussderconv(inpic, 2, 2, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lxyyy':
            outpic = gaussderconv(inpic, 1, 3, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        elif njetfcn == 'Lyyyy':
            outpic = gaussderconv(inpic, 0, 4, sigma, \
                                  gaussdernormmethod, gamma, epsilon)

        else:
            raise ValueError(f'NJetFcn {njetfcn} not implemented')

    return outpic


def make1Dgaussderfilter(
        order : int,
        sigma : float,
        N : int,
        gaussdermethod : str = 'samplgaussder'
    ) -> np.array :
    """Generates a mask for discrete approximation of a Gaussian derivative
    operator of a given order and at a given scale sigma by separable 
    filtering, using either of the methods:

    'samplgaussder'     - the sampled Gaussian derivative kernel with variance-based 
                          scale normalization
    'intgaussder'       - the integrated Gaussian derivative kernel with 
                          variance-based scale normalization
    'discgaussder'      - discrete derivative approximations applied to the discrete 
                          analogue of the Gaussian kernel

    The different discretization methods have the following relative advantages (+)
    and disadvantages (-):

    'samplgaussder': + no added scale offset in the spatial discretization
                     - for small values of sigma, the discrete kernel values may sum up 
                       to a value larger than the integral of the corresponding
                       continuous kernel
                     - for very small values of sigma, the kernels have a too 
                       narrow shape

    'intgaussder': + the discrete kernel values may sum up to a value close to the
                     L1-norm of the the continuous kernel over an infinite domain
                   - the box integration introduces a scale offset of 1/12 at 
                     coarser scales
                         
    'discgaussder': + the discrete kernels obey discrete scale-space properties
                    + the kernels obey an exact cascade smoothing property over
                      scales
                    + the computations of discrete derivative approximations of
                      different orders using small support discrete derivative
                      approximation masks requires substantially less computations
                      than convolutions with explicit Gaussian derivative kernels
                      of substantially larger support
                    
    The parameter N should specify the requested truncation bound for
    the filter for |x| > N, where N has to be determined in a complementary
    manner given some bound epsilon on the truncation error, for the given
    order of differentiation and the given scale value sigma.
    """
    if gaussdermethod == 'samplgaussder':
        return make1Dsamplgaussderfilter(order, sigma, N)

    if gaussdermethod == 'intgaussder':
        return make1Dintgaussderfilter(order, sigma, N)

    if gaussdermethod == 'discgaussder':
        return make1Ddiscgaussderfilter(order, sigma, N)

    if gaussdermethod == 'samplgaussdiff':
        return make1Dsamplgaussdifffilter(order, sigma, N)

    if gaussdermethod == 'normsamplgaussdiff':
        return make1Dnormsamplgaussdifffilter(order, sigma, N)
    
    if gaussdermethod == 'intgaussdiff':
        return make1Dintgaussdifffilter(order, sigma, N)

    raise ValueError(f"Gaussian derivative discretization method \
{gaussdermethod} not implemented")


def contdergaussderspread(
        order : int,
        sigma : float
    ) -> float :
    """Returns the spread measure corresponding to the spatial standard deviation
    of the absolute value of the Gaussian derivative kernel of a given order and
    for a given scale parameter sigma in units of the standard deviation of the
    Gaussian kernel.
    """
    if order == 0:
        return sigma

    if order == 1:
        return sqrt(2)*sigma

    if order == 2:
        return (exp(1) * pi / 2)**(1/4) \
               * sqrt(1 + 3 * sqrt(2 / (exp(1) * pi)) - 2 * erf(1 / sqrt(2))) \
               * sigma

    if order == 3:
        return sqrt((28 - 2 * exp(3/2)) / (4 + exp(3/2))) * sigma

    if order == 4:
        # Auto-generated code from Mathematica
        # E = exp(1)
        # const = (3*pow(27 + 11*sqrt(6),0.25)*sqrt((5880*sqrt(2) + 4801*sqrt(3) + \
        #    267*sqrt(485 + 198*sqrt(6)) + 109*sqrt(6*(485 + 198*sqrt(6))) + \
        #    (27 + 11*sqrt(6) + 267*sqrt(49 - 20*sqrt(6)) \
        #      + 109*sqrt(6*(49 - 20*sqrt(6))))* \
        #    pow(E,sqrt(6))) / \
        #    (5 + 2*sqrt(6) + sqrt(49 + 20*sqrt(6)) + \
        #    (sqrt(2) + sqrt(3) + sqrt(5 + 2*sqrt(6)))*pow(E,sqrt(6))))*sigma)/ \
        #    pow(3 + sqrt(6),2.75)
        # return const * sigma
        #
        # ==>> The above code causes a bug in Python. Replacing with numerical value
        return 1.48122 * sigma

    raise ValueError(f"Not implemented for order {order}")
