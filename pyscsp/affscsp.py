"""Affine Scale-Space and Scale-Space Derivative Toolbox for Python

For computing affine Gaussian kernels and affine Gaussian directional 
derivative kernels, as well as providing a computationally reasonably 
efficient way to compute filter banks of directional derivative responses 
over different orders of spatial differentiation, by defining directional
derivative approximation masks of small spatial support, which are to
be applied to image data that has been already smoothed by regular 
affine Gaussian kernels.

References:

Lindeberg (1993) Scale-Space Theory in Computer Vision, Springer.

Lindeberg and Garding (1997) "Shape-adapted smoothing in estimation 
of 3-D depth cues from affine distortions of local 2-D structure",
Image and Vision Computing 15: 415-434

Lindeberg (2013) "A computational theory of visual receptive fields", 
Biological Cybernetics, 107(6): 589-635.

Lindeberg (2021) "Normative theory of visual receptive fields", 
Heliyon 7(1): e05897: 1-20.

Relations between the scientific papers and concepts in this code:

Chapter 14 in the book (Lindeberg 1993) and the article 
(Lindeberg and Garding 1997) describe the notion of affine Gaussian
scale space, with its closedness property under affine image
transformations, referred to as affine covariance or affine equivariance.

The articles (Lindeberg 2013) and (Lindeberg 2021) demonstrate how
the spatial component of the receptive fields of simple cells in
the primary visual cortex can be well modelled by directional
derivatives of affine Gaussian kernels. In the code below, we
provide functions for generating such kernels, corresponding to
directional derivatives of affine Gaussian kernels and for computing
the effect of convolving images with such kernels.
"""

from math import exp, pi, sqrt, cos, sin
import numpy as np
from scipy.ndimage import correlate
from pyscsp.discscsp import dirdermask
from pyscsp.gaussders import N_from_epsilon_2D


def CxxCxyCyyfromlambda12phi(
        lambda1 : float,
        lambda2 : float,
        phi : float
    ) -> (float, float, float) :
    """Computes the parameters of spatial covariance matrix Sigma

    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    given a specification of the eigenvalues lambda1 and lambda2 of the
    covariance matrix as well as its orientation.

    Reference:

    Lindeberg (2013) "A computational theory of visual receptive fields", 
    Biological Cybernetics, 107(6): 589-635. (See Equation (68).)
    """
    Cxx = lambda1 * cos(phi)**2 + lambda2 * sin(phi)**2
    Cxy = (lambda1 - lambda2) * cos(phi) * sin(phi)
    Cyy = lambda1 * sin(phi)**2 + lambda2 * cos(phi)**2

    return Cxx, Cxy, Cyy


def CxxCxyCyyfromsigma12phi(
        sigma1 : float,
        sigma2 : float,
        phi : float
    ) -> (float, float, float) :
    """Computes the parameters of spatial covariance matrix Sigma

    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    given a specification of the eigenvalues scale parameter sigma1
    and sigma2 (the square roots of the eigenvalues) of the covariance 
    matrix as well as its orientation.

    Reference:

    Lindeberg (2013) "A computational theory of visual receptive fields", 
    Biological Cybernetics, 107(6): 589-635. (See Equation (68).)
    """
    lambda1 = sigma1**2
    lambda2 = sigma2**2

    return CxxCxyCyyfromlambda12phi(lambda1, lambda2, phi)


def sampldirderaffgausskernelfromlambda12phi(
        lambda1 : float,
        lambda2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        N : int
    ) -> np.ndarray :
    """Computes a kernel of size N x N representing the sampled directional
    derivative of order phiorder in the direction phi and of order orthorder
    in a direction orthogonal to phi.

    The kernel is defined as

    D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent the partial derivative operators in the 
    directions phi and its orthogonal direction orth, respectively. 

    The Gaussian kernel is, in turn, defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the spatial covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    represented by the parameterization

    Cxx = lambda1 * cos(phi)**2 + lambda2 * sin(phi)**2
    Cxy = (lambda1 - lambda2) * cos(phi) * sin(phi)
    Cyy = lambda1 * sin(phi)**2 + lambda2 * cos(phi)**2

    Note: You have to determine an appropriate choice of N in a complementary way.

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (23)).
    """
    # Generate a grid of spatial coordinates
    xbase = np.linspace(-N, N, 2*N + 1)
    ybase = np.linspace(-N, N, 2*N + 1)
    ybase = - ybase
    x, y = np.meshgrid(xbase, ybase, indexing='xy')

    # The code below has been autogenerated from Mathematica to C, then first
    # ported to Matlab by semi-automatic editing, and then further ported to
    # Python by another round of editing.
    # Therefore, some of the constructions may seem a bit odd ...
    E = exp(1)

    if (phiorder == 0) and (orthorder == 0):
        return 1 / (2 * np.power(E, \
                                 ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                                  (lambda1 * x**2 + lambda2 * y**2) * sin(phi)**2 - \
                                  (lambda1 - lambda2) * x * y * sin(2*phi)) \
                                 / (2 * lambda1 * lambda2)) \
                       * sqrt(lambda1 * lambda2) * pi)

    if (phiorder == 1) and (orthorder == 0):
        return - (lambda2 * (x * cos(phi) + y * sin(phi))) / \
                 (2 * np.power(E, \
                               ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                                (lambda1 * x**2 + lambda2 * y**2) *sin(phi)**2 - \
                                (lambda1 - lambda2) * x * y *sin(2*phi)) \
                               / (2 * lambda1 * lambda2))  \
                    * pow(lambda1 *lambda2, 1.5) * pi)

    if (phiorder == 0) and (orthorder == 1):
        return (-2 * lambda1 * y * cos(phi) + 2 * lambda1 * x * sin(phi)) / \
               (4 * np.power(E, \
                            ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                             (lambda1 * x**2 + lambda2 * y**2) * sin(phi)**2 - \
                             (lambda1 - lambda2) * x * y * sin(2*phi)) \
                            / (2 * lambda1 * lambda2)) \
                  * pow(lambda1 * lambda2, 1.5) * pi)

    if (phiorder == 2) and (orthorder == 0):
        return (-2 * lambda1 + x**2 + y**2 + (x**2 - y**2) * cos(2*phi) + \
                 2 * x * y * sin(2*phi)) / \
               (4 * np.power(E, \
                             ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                              (lambda1 * x**2 + lambda2 * y**2) * sin(phi)**2 - \
                              (lambda1 - lambda2) * x * y * sin(2*phi)) \
                             / (2 * lambda1 * lambda2)) \
                  * pow(lambda1, 2) * sqrt(lambda1 * lambda2) * pi)

    if (phiorder == 1) and (orthorder == 1):
        return - (lambda2 * (x * cos(phi) + y * sin(phi))) *\
               (-2 * lambda1 * y * cos(phi) + 2 * lambda1 * x * sin(phi)) / \
               (4 * np.power(E, \
                            ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                             (lambda1 * x**2 + lambda2 * y**2) * sin(phi)**2 - \
                             (lambda1 - lambda2) * x * y * sin(2*phi)) \
                            / (2 * lambda1 * lambda2)) \
                  * pow(lambda1 * lambda2, 1.5) * pi)

    if (phiorder == 0) and (orthorder == 2):
        return (-2 * lambda2 + x**2 + y**2 + (-x**2 + y**2)*cos(2*phi) \
                - 2 * x * y * sin(2*phi)) / \
                (4 * np.power(E, \
                              ((lambda2 * x**2 + lambda1 * y**2) * cos(phi)**2 + \
                               (lambda1 * x**2 + lambda2 * y**2) * sin(phi)**2 - \
                               (lambda1 - lambda2) * x * y * sin(2*phi)) \
                              /(2 * lambda1 * lambda2)) \
                    * pow(lambda2, 2) * sqrt(lambda1 * lambda2) * pi)

    raise ValueError(f"Not implemented for phiorder {phiorder} orthorder {orthorder}")


def sampldirderaffgausskernelfromsigma12phi(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        N : int
    ) -> np.ndarray :
    """Computes a kernel of size N x N representing the sampled directional
    derivative of order phiorder in the direction phi and of order orthorder
    in a direction orthogonal to phi.

    The kernel is defined as

    D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for .

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent the partial derivative operators in the 
    directions phi and and its orthogonal direction orth, respectively. 

    The Gaussian kernel is, in turn, defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the spatial covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    represented by the parameterization

    Cxx = sigma1^2 * cos(phi)**2 + sigma2^2 * sin(phi)**2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)**2 + sigma2^2 * cos(phi)**2

    Note: You have to determine an appropriate choice of N in a complementary way.

    References:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (23)).
    """
    lambda1 = sigma1**2
    lambda2 = sigma2**2

    return sampldirderaffgausskernelfromlambda12phi(lambda1, lambda2, phi, \
                                                    phiorder, orthorder, N)


def scnormsampldirderaffgausskernelfromsigma12phi(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        N : int
    ) -> np.ndarray :
    """Computes a kernel of size N x N representing the sampled directional
    derivative of order phiorder in the direction phi and of order orthorder
    in a direction orthogonal to phi.

    The kernel is defined as

    sigma1^phiorder sigma2^orthorder D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent the partial derivative operators in the 
    directions phi and and its orthogonal direction orth, respectively. 

    The Gaussian kernel is, in turn, defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the spatial covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    represented by the parameterization

    Cxx = sigma1^2 * cos(phi)**2 + sigma2^2 * sin(phi)**2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)**2 + sigma2^2 * cos(phi)**2

    Note: You have to determine an appropriate choice of N in a complementary way.

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (31)).
    """
    lambda1 = sigma1**2
    lambda2 = sigma2**2
    scalenormfactor = sigma1**phiorder * sigma2**orthorder

    return scalenormfactor * \
           sampldirderaffgausskernelfromsigma12phi(lambda1, lambda2, phi, \
                                                   phiorder, orthorder, N)


def numdirdersamplaffgausskernel(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        N : int
    ) -> np.ndarray :
    """Computes a kernel of size N x N representing a numerical approximation
    of the directional derivative of order phiorder in the direction phi and 
    of order orthorder in a direction orthogonal to phi.

    The kernel is defined as

    D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent (discrete approximations of) the partial 
    derivative operators in the directions phi and its orthogonal direction
    orth, respectively. 

    The Gaussian kernel is, in turn, defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the spatial covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    represented by the parameterization

    Cxx = sigma1^2 * cos(phi)**2 + sigma2^2 * sin(phi)**2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)**2 + sigma2^2 * cos(phi)**2

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (23)).
    """
    # ==>> Complement the following code by removal of boundary effects
    affgausskernel = samplaffgausskernel(sigma1, sigma2, phi, N)
    mask = dirdermask(phi, phiorder, orthorder)

    return correlate(affgausskernel, mask)


def samplaffgausskernel(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        N : int
    ) -> np.ndarray :
    """Computes a sampled affine Gaussian kernel of size N x N defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    parameterized as

    Cxx = sigma1^2 * cos(phi)^2 + sigma2^2 * sin(phi)^2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)^2 + sigma2^2 * cos(phi)^2

    References:

    Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.

    Lindeberg and Garding (1997) "Shape-adapted smoothing in estimation 
    of 3-D depth cues from affine distortions of local 2-D structure",
    Image and Vision Computing 15:415-434
    """
    return sampldirderaffgausskernelfromsigma12phi(sigma1, sigma2, phi, 0, 0, N)


def samplaffgaussconv(
        inpic,
        sigma1 : float,
        sigma2 : float,
        phi : float,
        epsilon : int = 0.0001
    ) -> np.ndarray :
    """Convolves the input image inpic sampled affine Gaussian kernel defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    parameterized as

    Cxx = sigma1^2 * cos(phi)^2 + sigma2^2 * sin(phi)^2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)^2 + sigma2^2 * cos(phi)^2
    
    and given a bound epsilon on the truncation error at the boundaries,
    if the smoothing kernel is truncated at the tails.
    """
    # Estimate a size for truncating the Gaussian derivative kernels
    N = N_from_epsilon_2D(0, max(sigma1, sigma2), epsilon)

    affgaussfilter = samplaffgausskernel(sigma1, sigma2, phi, N)

    return correlate(inpic, affgaussfilter)


def applyaffdirder(
        smoothpic,
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        normdermethod = 'varnorm'
    ) -> np.ndarray :
    """Applies a directional derivative operator of order phiorder in the direction
    phi and of order orthorder in the orthogonal direction, to an image that is
    assumed to have been smoothed with an affine Gaussian kernel with standard
    deviation sigma1 in the direction phi and standard deviation orthorder
    in the orthogonal direction, where the parameter normdermethod specifies
    the type of scale normalization to be used for the affine Gaussian derivatives.
    """
    if normdermethod == 'varnorm':
        mask = scnormaffdirdermask(sigma1, sigma2, phi, phiorder, orthorder)
    elif normdermethod == 'nonormalization':
        mask = dirdermask(phi, phiorder, orthorder)
    else:
        raise ValueError(f"Scale normalization method {normdermethod} not implemented")

    return correlate(smoothpic, mask)


def scnormaffdirdermask(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int
    ) -> np.ndarray :
    """Returns a discrete directional derivative approximation mask, such that
    application of this mask to an image smoothed by a zero-order affine Gaussian 
    kernel (assumed to have been determined using the same values of sigma1,
    sigma2 and phi) gives an approximation of the scale-normalized directional 
    derivative according to

    sigma1^phiorder sigma2^orthorder D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent the partial derivative operators in the 
    directions phi and its orthogonal direction orth, respectively, where it
    is assumed that convolution with g(x; Sigma) is computed outside of this 
    function, and using the same values of sigma1, sigma2 and phi.

    The intention is that the mask returned by this function should be applied
    to affine Gaussian smoothed images. Specifically, for an image processing
    method that makes use of a filter bank of directional derivatives of 
    affine Gaussian kernels, the intention is that the computationally heavy
    affine Gaussian smoothing operation should be performed only once, and
    that different directional derivative approximation masks should then
    be applied to the same affine Gaussian smoothed image, thus saving
    a substantial amount of work, compared to applying full size affine
    Gaussian directional derivative masks for different choices of orders
    of the directional derivatives.

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (31)).
    """
    scalenormfactor = sigma1**phiorder * sigma2**orthorder
    rawmask = dirdermask(phi, phiorder, orthorder)

    return scalenormfactor * rawmask


def scnormnumdirdersamplaffgausskernel(
        sigma1 : float,
        sigma2 : float,
        phi : float,
        phiorder : int,
        orthorder : int,
        N : int
    ) -> np.ndarray :
    """Computes a kernel of size N x N representing the sampled directional
    derivative of order phiorder in the direction phi and of order orthorder
    in a direction orthogonal to phi.

    The kernel is defined as

    sigma1^phiorder sigma2^orthorder D_phi^phiorder D_orth^orthorder g(x; Sigma)

    for

    D_phi  =  cos phi D_x + sin phi D_y
    D_orth = -sin phi D_x + cos phi D_y

    where D_phi and D_orth represent (discrete approximations of) the partial 
    derivative operators in the directions phi and its orthogonal direction orth, 
    respectively, and with the Gaussian kernel is, in turn, defined as

    g(x; Sigma) = 1/(2 * pi * det Sigma) * exp(-x^T Sigma^(-1) x/2)

    with the spatial covariance matrix 
    
    Sigma = [[Cxx, Cxy],
             [Cxy, Cyy]]

    represented by the parameterization

    Cxx = sigma1^2 * cos(phi)**2 + sigma2^2 * sin(phi)**2
    Cxy = (sigma1^2 - sigma2^2) * cos(phi) * sin(phi)
    Cyy = sigma1^2 * sin(phi)**2 + sigma2^2 * cos(phi)**2

    Note: The intention is not that this function should be used for computing
    output from receptive field responses. It is mererly intended for purposes
    of graphical illustration of receptive fields.

    Reference:

    Lindeberg (2021) "Normative theory of visual receptive fields", 
    Heliyon 7(1): e05897: 1-20. (See Equation (31)).
    """
    # ==>> Complement the following code by removal of boundary effects
    affgausskernel = samplaffgausskernel(sigma1, sigma2, phi, N)
    scnormmask = scnormaffdirdermask(sigma1, sigma2, phi, phiorder, orthorder)

    return correlate(affgausskernel, scnormmask)
