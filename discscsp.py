import numpy as np
from scipy.ndimage import correlate1d
from scipy.ndimage import correlate
from scipy.special import ive
from math import sqrt
from scipy.special import erf
from scipy.special import erfcinv
from math import exp
from math import pi
from typing import NamedTuple


# Discrete Scale Space and Scale-Space Derivative Toolbox for Python
#
# For computing discrete scale-space smoothing by convolution with the discrete
# analogue of the Gaussian kernel and for computing discrete derivative approximations
# by applying central difference operators to the smoothed data. Then, different
# types of feature detectors can be defined by combining discrete analogues of the
# Gaussian derivative operators into differential expressions.
#
# This code is the result of porting a subset of the routines in the Matlab packages
# discscsp and discscspders to Python.
#
# Note: The scale normalization does not explicitly compensate for the additional 
# variance 1/12 for the integrated Gaussian kernel or the additional variance 1/6
# for the linearly integrated Gaussian kernel.
#
# References:
#
# Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
# Pattern Analysis and Machine Intelligence, 12(3): 234--254.
#
# Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
# A basis for low-level feature detection", Journal of Mathematical Imaging and Vision, 
# 3(4): 349-376.
#
# Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
#
# Lindeberg (1998) "Feature detection with automatic scale selection", 
# International Journal of Computer Vision, vol 30(2): 77-116.
#
# Lindeberg (1998) "Edge detection and ridge detection with automatic scale selection", 
# International Journal of Computer Vision, vol 30(2): 117-154.

# Compared to the original Matlab code, the following implementation is reduced in the following ways:
# - there is no handling of scale normalization powers gamma that are not equal to one
# - Lp-normalization is only implemented for p = 1
# - much fewer functions of the N-jet have so far been implemented
# - there is no passing of additional parameters to functions of the N-jet
# - this reimplementation has not yet been thoroughly tested


def scspconv(inpic, sigma, scspmethod='discgauss', epsilon=0.00000001):
    if (isinstance(scspmethod, str)):
        scspmethodname = scspmethod;
    else:
        scspmethodname = scspmethod.methodname
        epsilon = scspmethod.epsilon

    if (scspmethodname == 'discgauss'):
        outpic = discgaussconv(inpic, sigma, epsilon)
    elif (scspmethodname == 'samplgauss'):
        outpic = samplgaussconv(inpic, sigma, epsilon)
    elif (scspmethodname == 'intgauss'):
        outpic = intgaussconv(inpic, sigma, epsilon)        
    elif (scspmethodname == 'linintgauss'):
        outpic = linintgaussconv(inpic, sigma, epsilon)        
    else:
        raise ValueError('Scale space method %s not implemented' % scspmethodname)

    return outpic

        
def discgaussconv(inpic, sigma, epsilon=0.00000001):
    ndim = inpic.ndim
    sep1Dfilter = make1Ddiscgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(inpic, sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(inpic, sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for l in range(0, inpic.shape[2]):
            outpic[:, :, l] = discgaussconv(inpic[:, :, l], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Ddiscgaussfilter(sigma, epsilon=0.00000001, D=1):
    s = sigma*sigma
    tmpvecsize = np.ceil(1 + 1.5*gaussfiltsize(sigma, epsilon, D))
    # Generate filter coefficients from modified Bessel functions
    longhalffiltvec = ive(np.arange(0, tmpvecsize+1), s)
    halffiltvec = truncfilter(longhalffiltvec, truncerrtransf(epsilon, D))
    filtvec = mirrorhfilter(halffiltvec);
    return filtvec


def samplgaussconv(inpic, sigma, epsilon=0.00000001):
    ndim = inpic.ndim
    sep1Dfilter = make1Dsamplgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(inpic, sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(inpic, sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for l in range(0, inpic.shape[2]):
            outpic[:, :, l] = samplgaussconv(inpic[:, :, l], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dsamplgaussfilter(sigma, epsilon=0.00000001, D=1):
    vecsize = np.ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    return gauss(x, sigma);


def gauss(x, sigma=1.0):
    return 1/(sqrt(2*pi)*sigma)*np.exp(-(x**2/(2*sigma**2)));


def intgaussconv(inpic, sigma, epsilon=0.00000001):
    ndim = inpic.ndim
    sep1Dfilter = make1Dintgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(inpic, sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(inpic, sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for l in range(0, inpic.shape[2]):
            outpic[:, :, l] = intgaussconv(inpic[:, :, l], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dintgaussfilter(sigma, epsilon=0.00000001, D=1):
    # Box integrated Gaussian kernel over each pixel support region
    # Remark: Adds additional spatial variance 1/12 to the kernel
    vecsize = np.ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    return scaled_erf(x + 0.5, sigma) - scaled_erf(x - 0.5, sigma)


def scaled_erf(x, sigma=1.0):
    return 1/2*(1 + erf(x/(sqrt(2)*sigma)))


def linintgaussconv(inpic, sigma, epsilon=0.00000001):
    ndim = inpic.ndim
    sep1Dfilter = make1Dlinintgaussfilter(sigma, epsilon, ndim)

    if (ndim == 1):
        outpic = correlate1d(inpic, sep1Dfilter)
    elif (ndim == 2):
        tmppic = correlate1d(inpic, sep1Dfilter, 0)
        outpic = correlate1d(tmppic, sep1Dfilter, 1)
    elif (ndim == 3):
        # Treat as multilayer image
        outpic = np.zeros(inpic.shape)
        for l in range(0, inpic.shape[2]):
            outpic[:, :, l] = linintgaussconv(inpic[:, :, l], sigma, epsilon)
    else:
        raise ValueError('Cannot handle images of dimensionality %d' % ndim)
    
    return outpic


def make1Dlinintgaussfilter(sigma, epsilon=0.00000001, D=1):
    # Linearly integrated Gaussian kernel over each extended pixel support region 
    # Remark: Adds additional spatial variance 1/6 to the kernel
    vecsize = np.ceil(1.1*gaussfiltsize(sigma, epsilon, D))
    x = np.linspace(-vecsize, vecsize, 1+2*vecsize)
    # The following equation is the result of a closed form integration of the expression
    # for the filter coefficients in Eq (2.89) on page 52 in Lindeberg's PhD thesis
    return x_scaled_erf(x + 1, sigma) - 2*x_scaled_erf(x, sigma) + x_scaled_erf(x - 1, sigma) + \
           sigma**2 * (gauss(x + 1, sigma) - 2*gauss(x, sigma) + gauss(x - 1, sigma))


def x_scaled_erf(x, sigma=1.0):
    return x * scaled_erf(x, sigma)


def gaussfiltsize(sigma, epsND, D):
    s = sigma*sigma
    eps1D = truncerrtransf(epsND, D)
    N = sqrt(2*s)*erfcinv(eps1D)    
    return N


def truncerrtransf(epsND, D):
    eps1D = 1 - (1 - epsND)**(1/D)
    return eps1D


def truncfilter(longhalffilter, epsilon):
    length = longhalffilter.shape[0]
    sum = longhalffilter[0]
    
    i = 1
    while ((sum < 1-epsilon) and (i < length)):
        sum = sum + 2*longhalffilter[i]
        i += 1

    return longhalffilter[0:i]


def mirrorhfilter(halffilter):
    length = halffilter.shape[0]
    revfilter = halffilter[::-1]
    return np.append(revfilter[0:length-1], halffilter)


def deltafcn(xsize, ysize):
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


def dxmask():
    return np.array([[-1/2, 0, 1/2]])


def dymask():
    return np.array([[+1/2], \
                     [   0], \
                     [-1/2]])


def dxxmask():
    return np.array([[1, -2, 1]])


def dxymask():
    return np.array([[-1/4, 0, +1/4], \
                     [   0, 0,    0], \
                     [+1/4, 0, -1/4]])


def dyymask():
    return np.array([[+1], \
                     [-2], \
                     [+1]])


def dxxxmask():
    return np.array([[-1/2, 1, 0, -1, 1/2]])


def dxxymask():
    return np.array([[+1/2, -1, +1/2], \
                     [   0,  0,    0], \
                     [-1/2, +1, -1/2]])


def dxyymask():
    return np.array([[-1/2, 0, +1/2], \
                     [  +1, 0,   -1], \
                     [-1/2, 0, +1/2]])


def dyyymask():
    return np.array([[+1/2], \
                     [  -1], \
                     [   0], \
                     [  +1], \
                     [-1/2]])


def dxxxxmask():
    return np.array([[1, -4, 6, -4, 1]])


def dxxxymask():
    return np.array([[-1/4, +1/2, 0, -1/2, +1/4], \
                     [   0,    0, 0,    0,    0], \
                     [+1/4, -1/2, 0, +1/2, -1/4]])


def dxxyymask():
    return np.array([[+1, -2, +1], \
                     [-2, +4, -2], \
                     [+1, -2, +1]])


def dxyyymask():
    return np.array([[-1/4, 0, +1/4], \
                     [+1/2, 0, -1/2], \
                     [   0, 0,    0], \
                     [-1/2, 0, +1/2], \
                     [+1/4, 0, -1/4]])


def dyyyymask():
    return np.array([[+1], \
                     [-4], \
                     [+6], \
                     [-4], \
                     [+1]])


def computeNjetfcn(inpic, njetfcn, sigma, normdermethod='discgaussLp'):
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    smoothpic = scspconv(inpic, sigma, normdermethod.scspmethod)
    return applyNjetfcn(smoothpic, njetfcn, sigma, normdermethod)


def applyNjetfcn(smoothpic, njetfcn, sigma=1.0, normdermethod='discgaussLp'):
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)
        
    if ((smoothpic.ndim == 3) and (smoothpic.shape[2] > 1)):
        numlayers = smoothpic.shape[2]
        outpic = np.zeros(smoothpic.shape)
        for l in range(0, numlayers):
            outpic[:, :, l] = applyNjetfcn(smoothpic[:, :, l], njetfcn, sigma, normdermethod)
    else:
        if (njetfcn == 'L'):
            outpic = smoothpic
        elif (njetfcn == 'Lx'):
            outpic = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())
        elif (njetfcn == 'Ly'):
            outpic = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
        elif (njetfcn == 'Lxx'):
            outpic = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
        elif (njetfcn == 'Lxy'):
            outpic = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
        elif (njetfcn == 'Lyy'):
            outpic = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
        elif (njetfcn == 'Lv'):
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())   
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            outpic = np.sqrt(Lx*Lx + Ly*Ly)
        elif (njetfcn == 'Lv2'):
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())   
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            outpic = Lx*Lx + Ly*Ly
        elif (njetfcn == 'Laplace'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())   
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            outpic = Lxx + Lyy
        elif (njetfcn == 'detHessian'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            outpic = Lxx*Lyy - Lxy*Lxy
        elif (njetfcn == 'sqrtdetHessian'):
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            detHessian = Lxx*Lyy - Lxy*Lxy
            outpic = np.sign(detHessian) * np.sqrt(np.abs(detHessian))
        elif (njetfcn == 'Lv2Lvv'):
            # 2nd-order derivative in gradient direction (used for edge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            outpic = Lx*Lx*Lxx + 2*Lx*Ly*Lxy + Ly*Ly*Lyy;
        elif (njetfcn == 'Lv3Lvvv'):
            # 3rd-order derivative in gradient direction (used for edge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            Lxxx = normderfactor(3, 0, sigma, normdermethod) * correlate(smoothpic, dxxxmask())
            Lxxy = normderfactor(2, 1, sigma, normdermethod) * correlate(smoothpic, dxxymask())
            Lxyy = normderfactor(1, 2, sigma, normdermethod) * correlate(smoothpic, dxyymask())
            Lyyy = normderfactor(0, 3, sigma, normdermethod) * correlate(smoothpic, dyyymask())
            outpic = Lx*Lx*Lx*Lxxx + 3*Lx*Lx*Ly*Lxxy + 3*Lx*Ly*Ly*Lxyy + Ly*Ly*Ly*Lyyy;
        elif (njetfcn == 'Lp'):
            # 1st-order derivative in principal curvature direction (used for ridge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = sinbeta * Lx - cosbeta * Ly
        elif (njetfcn == 'Lq'):
            # 1st-order derivative in principal curvature (used for ridge detection)
            Lx = normderfactor(1, 0, sigma, normdermethod) * correlate(smoothpic, dxmask())
            Ly = normderfactor(0, 1, sigma, normdermethod) * correlate(smoothpic, dymask())
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            tmp = (Lxx - Lyy) /(np.finfo(float).eps + np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))
            cosbeta = np.sqrt((1 + tmp)/2)
            sinbeta = np.sign(Lxy) * np.sqrt((1 - tmp)/2)
            outpic = cosbeta * Lx + sinbeta * Ly;
        elif (njetfcn == 'Lpp'):
            # 2nd-order derivative in principal curvature direction 
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            outpic = ((Lxx + Lyy) - np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2
        elif (njetfcn == 'Lqq'):
            # 2nd-order derivative in principal curvature direction 
            Lxx = normderfactor(2, 0, sigma, normdermethod) * correlate(smoothpic, dxxmask())
            Lxy = normderfactor(1, 1, sigma, normdermethod) * correlate(smoothpic, dxymask())
            Lyy = normderfactor(0, 2, sigma, normdermethod) * correlate(smoothpic, dyymask())
            outpic = ((Lxx + Lyy) + np.sqrt((Lxx - Lyy)*(Lxx - Lyy) + 4*Lxy*Lxy))/2
        elif (njetfcn == 'Lxxx'):
            outpic = normderfactor(3, 0, sigma, normdermethod) * correlate(smoothpic, dxxxmask())
        elif (njetfcn == 'Lxxy'):
            outpic = normderfactor(2, 1, sigma, normdermethod) * correlate(smoothpic, dxxymask())
        elif (njetfcn == 'Lxyy'):
            outpic = normderfactor(1, 2, sigma, normdermethod) * correlate(smoothpic, dxyymask())
        elif (njetfcn == 'Lyyy'):
            outpic = normderfactor(0, 3, sigma, normdermethod) * correlate(smoothpic, dyyymask())
        elif (njetfcn == 'Lxxxx'):
            outpic = normderfactor(4, 0, sigma, normdermethod) * correlate(smoothpic, dxxxxmask())
        elif (njetfcn == 'Lxxxy'):
            outpic = normderfactor(3, 1, sigma, normdermethod) * correlate(smoothpic, dxxxymask())
        elif (njetfcn == 'Lxxyy'):
            outpic = normderfactor(2, 2, sigma, normdermethod) * correlate(smoothpic, dxxyymask())
        elif (njetfcn == 'Lxyyy'):
            outpic = normderfactor(1, 3, sigma, normdermethod) * correlate(smoothpic, dxyyymask())
        elif (njetfcn == 'Lyyyy'):
            outpic = normderfactor(0, 4, sigma, normdermethod) * correlate(smoothpic, dyyyymask())
        else:
            raise ValueError('NJetFcn %s not implemented yet' % njetfcn)

    return outpic


def normderfactor(xorder, yorder, sigma, normdermethod):
    if (isinstance(normdermethod, str)):
        normdermethod = defaultscspnormdermethodobject(normdermethod)

    if (normdermethod.normdermethod == 'none'):
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
                (discgaussder1D_L1norm(xorder, sigma, normdermethod.scspmethod.epsilon) * \
                discgaussder1D_L1norm(yorder, sigma, normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'samplgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (samplgaussder1D_L1norm(xorder, sigma, normdermethod.scspmethod.epsilon) * \
                samplgaussder1D_L1norm(yorder, sigma, normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'intgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (intgaussder1D_L1norm(xorder, sigma, normdermethod.scspmethod.epsilon) * \
                intgaussder1D_L1norm(yorder, sigma, normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented for gamma = 1.0')
        elif (normdermethod.scspmethod.methodname == 'linintgauss'):
            if (normdermethod.gamma == 1.0):
                value = \
                (normgaussder1D_L1norm(xorder, sigma) * \
                normgaussder1D_L1norm(yorder, sigma)) / \
                (linintgaussder1D_L1norm(xorder, sigma, normdermethod.scspmethod.epsilon) * \
                linintgaussder1D_L1norm(yorder, sigma, normdermethod.scspmethod.epsilon))
            else:
                raise ValueError('Lp-normalization so far only implemented for gamma = 1.0')
        else:
            raise ValueError('Lp-normalization not implemented for scale-space derivative method %s' \
                                 % normdermethod.scspmethod.methodname)
    else:
        raise ValueError('Derivative method %s not implemented yet' % normdermethod.normdermethod)

    return value


def normgaussder1D_L1norm(order, sigma, gammapar=1.0):
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


def discgaussder1D_L1norm(order, sigma, epsilon=0.00000001):
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


def samplgaussder1D_L1norm(order, sigma, epsilon=0.00000001):
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


def intgaussder1D_L1norm(order, sigma, epsilon=0.00000001):
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


def linintgaussder1D_L1norm(order, sigma, epsilon=0.00000001):
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

        
class ScSpMethod(NamedTuple):
    methodname: str # either 'discgauss', 'samplgauss', 'intgauss' or 'linintgauss'
    epsilon: float


def discgaussmethod(epsilon):
    return ScSpMethod('discgauss', epsilon)


class ScSpNormDerMethod(NamedTuple):
    scspmethod: ScSpMethod
    normdermethod: str # either 'none', 'varnorm' or 'Lpnorm'
    gamma: float


def scspnormdermethodobject(scspmethod='discgauss', normdermethod='Lpnorm', gamma=1.0, epsilon=0.00000001):
    return ScSpNormDerMethod(ScSpMethod(scspmethod, epsilon), normdermethod, gamma)


def defaultscspnormdermethodobject(scspnormdermethod='discgaussLp', gamma=1.0):
    if (scspnormdermethod == 'discgauss'):
        object = scspnormdermethodobject('discgauss', 'none', gamma)
    elif (scspnormdermethod == 'discgaussvar'):
        object = scspnormdermethodobject('discgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'discgaussLp'):
        object = scspnormdermethodobject('discgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'samplgauss'):
        object = scspnormdermethodobject('samplgauss', 'none', gamma)
    elif (scspnormdermethod == 'samplgaussvar'):
        object = scspnormdermethodobject('samplgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'samplgaussLp'):
        object = scspnormdermethodobject('samplgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'intgauss'):
        object = scspnormdermethodobject('intgauss', 'none', gamma)
    elif (scspnormdermethod == 'intgaussvar'):
        object = scspnormdermethodobject('intgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'intgaussLp'):
        object = scspnormdermethodobject('intgauss', 'Lpnorm', gamma)
    elif (scspnormdermethod == 'linintgauss'):
        object = scspnormdermethodobject('linintgauss', 'none', gamma)
    elif (scspnormdermethod == 'linintgaussvar'):
        object = scspnormdermethodobject('linintgauss', 'varnorm', gamma)
    elif (scspnormdermethod == 'linintgaussLp'):
        object = scspnormdermethodobject('linintgauss', 'Lpnorm', gamma)
    else:
        error('Scale-space derivative method %s not implemented yet' % scspnormdermethod)
    return object


def variance(filter):
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

    xvar = x2mom - xmean*xmean
    yvar = y2mom - ymean*ymean
    
    return [[x2mom - xmean*xmean, xymom - xmean*ymean], \
            [xymom - xmean*ymean, y2mom - ymean*ymean]]


def filtermean(filter):
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

    
if __name__ == '__main__': 
    main() 
