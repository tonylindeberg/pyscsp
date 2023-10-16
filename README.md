# pyscsp : Scale-Space Toolbox for Python

Contains the following modules:

## discscsp: Discrete Scale-Space and Scale-Space Derivative Toolbox for Python:

This module comprises:

-- functions for computing spatial scale-space representations by spatial smoothing 
     with the discrete analogue of the Gaussian kernel or other discrete approximations 
	 of the continuous Gaussian kernel, that is used for defining a Gaussian 
	 scale-space representation.

-- functions for computing differential expressions in terms of scale-normalized
    Gaussian derivatives for different purposes in feature detection
    from image data, such as edge detection, interest point detection
    (blob detection or corner detection) and ridge detection.
	
For examples of how to apply these functions for computing scale-space
features, please see the enclosed Jupyter notebook 
[discscspdemo.ipynb](https://github.com/tonylindeberg/pyscsp/blob/main/discscspdemo.ipynb).

For more technical descriptions about the respective functions, as well
as explanations of the theoretical properties for different discrete
approximations of the Gaussian kernel, please see the documentation
strings for the respective functions in the source code in discscsp.py.

## affscsp: Affine Scale-Space and Scale-Space Derivative Toolbox for Python

This module comprises.

-- functions for computing affine Gaussian kernels and affine Gaussian directional kernels. 

-- functions for a computationally reasonably efficient way to compute 
filter banks of directional derivative responses over different orders of 
spatial differentiation.

For more technical descriptions about the respective functions, please
see the documentation strings for the respective functions in the source
code in affscsp.py.

## torchscsp: Subset of functionalities for use in PyTorch:

-- functions for generating 1-D discrete approximations of the Gaussian kernel
     for spatial smoothing with separable filtering in PyTorch.
	 
-- discrete derivative approximation masks for computing discrete approximations
     of Gaussian derivatives and Gaussian derivative layers in PyTorch.

## Installation

These modules can be installed using pip.

To install only the discscsp and affscsp modules
(without installing the torchscsp module which requires a larger
installation of PyTorch) do:
```bash
pip install pyscsp
```

To install also the torchscsp module, do instead perform the
following command:
```bash
pip install 'pyscsp[torch]'
```
Note, however, that you must then have PyTorch already installed to use
this option. Otherwise, the installation command will generate an
error message.

These modules can also be downloaded directly from GitHub:

```bash
git clone git@github.com:tonylindeberg/pyscsp.git
```


## References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.
([preprint](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-58057))

Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
A basis for low-level feature detection", Journal of Mathematical Imaging and Vision, 
3(4): 349-376.
([preprint](https://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A473368&dswid=3752))

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.
([Online edition](http://dx.doi.org/10.1007/978-1-4757-6465-9))

Lindeberg (1998a) "Feature detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 77-116.
([preprint](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-40224))

Lindeberg (1998b) "Edge detection and ridge detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 117-154.
([preprint](http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-40226))

Lindeberg (2009) "Scale-space". In: B. Wah (Ed.) Wiley Encyclopedia of Computer 
Science and Engineering, John Wiley & Sons, pp. 2495-2504.
([preprint](https://kth.diva-portal.org/smash/record.jsf?pid=diva2%3A441147&dswid=2409))

Lindeberg (2015) "Image matching using generalized scale-space
interest points", Journal of Mathematical Imaging and Vision, 52(1):
3-36.
([Open Access](https://dx.doi.org/10.1007/s10851-014-0541-0))

Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.
([Open Access](https://doi.org/10.1007/s10851-021-01057-9))

## Relations between the scientific papers and concepts in this code

The paper (Lindeberg 1990) describes the discrete analogue of the
Gaussian kernel used for discrete implementation of Gaussian
smoothing, including its theoretical properties and how it can be
defined by uniqueness from a set of theoretical assumptions
(scale-space axioms) that reflect desirable properties of a
scale-space smoothing operation. This paper also describes some of the
theoretical properties of the sampled Gaussian kernel.

The paper (Lindeberg 1993a) describes how discrete derivative
approximations defined by applying difference operators to a discrete
scale-space representation preserve scale-space properties of discrete
approximations of Gaussian derivatives, provided that the scale-space
smoothing operattion is performed using the discrete analogue of the
Gaussian kernel.

Chapters 3-5 in (Lindeberg 1993) give a more extensive treatment of
discrete scale-space representation defined by convolution with the
discrete analogue of the Gaussian kernel, including scale-space
properties of discrete derivative approximations defined by applying
difference operators to the discrete scale-space representation
defined by convolution with the discrete analogue of the Gaussian
kernel. This treatment also describes theoretical properties of the
sampled Gaussian kernel, the integrated Gaussian kernel and the
linearily integrated Gaussian kernel.

The paper (Lindeberg 1998a) describes the blob detector based on the
spatial extrema of the Laplacian operator (N-jet function 'Laplace'), the
interest point detector based on spatial extrema of the determinant of
the Hessian operator (N-jet function 'detHessian') and the corner
detector based on spatial extrema of the rescaled level curve
curvature operator (N-jet function 'Kappa'). This paper also defines
the notion of gamma-normalized scale-space derivatives by multiplying
the regular Gaussian derivative operators by the scale parameter s =
sigma^2 raised to the power of gamma multiplied by the order of
differentiation and divided by two, including a way to approximate this operator for 
discrete image data based on lp-normalization of the derivative 
operator, with the power p in the Lp-norms and lp-norms related to
the scale normalization power gamma.

The paper (Lindeberg 1998b) describes the differential definition of
edge detection from local directional derivatives of the image
intensity in the gradient direction (N-jet functions 'Lv', 'Lv2Lvv' and
Lv3Lvv') as well as corresponding ridge and valley detectors defined
from directional derivatives in the principal curvature directions (p,
q) of the grey-level landscape (N-jet functions 'Lp', 'Lq', 'Lpp' and
'Lqq').

The paper (Lindeberg 2009) gives a comprehensive overview of basic
components in scale-space theory, and can in this respect serve as a
good first introduction to this area, including demonstrations of how
different types of differential invariants in scale-space (in this
code referred to as N-jet functions) can be used for basic purposes of
detecting image features in image data.

The paper (Lindeberg 2015) gives a more modern treatment of some of
the concepts described in (Lindeberg 1998a), regarding the use of
spatial extrema of the Laplacian operator (N-jet function 'Laplace'),
the determinant of the Hessian operator (N-jet function 'detHessian')
and the rescaled level curve curvature operator (N-jet function 'Kappa')
for interest point detection.

The paper (Lindeberg 2022) defines the notion of a Gaussian derivative
layer, as a linear combination of scale-normalized Gaussian derivative
responses, as a basic concept for defining provably scale-covariant
and scale-invariant deep networks.
