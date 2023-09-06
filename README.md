# pyscsp
Scale-space functions in Python

Contains the following subpackages:

* discscsp.py: Discrete Scale Space and Scale-Space Derivative Toolbox for Python:

This package comprises:

-- functions for computing spatial scale-space representations by spatial smoothing 
     with the discrete analogue of the Gaussian kernel or other discrete approximations 
	 of the continuous Gaussian kernel that is used for defining a Gaussian 
	 scale-space representation.

-- functions for computing differential expressions in terms of scale-normalized
    Gaussian derivatives for different purposes in feature detection
    from image data, such as edge detection, interest point detection
    (blob detection or corner detection) and ridge detection.
	
For examples of how to apply these functions for computing scale-space
features, please see the enclosed Jupyter notebook discscspdemo.ipynb.

For more technical descriptions about the respective functions, as well
as explanations of the theoretical properties for different discrete
approximations of the Gaussian kernel, please see the documentation
strings for the respective functions in the source code in discscsp.py.

* torchscsp.py: Subset of functionalities for use in PyTorch:

-- functions for generating 1-D discrete approximations of the Gaussian kernel
     for spatial smoothing with separable filtering in PyTorch.
	 
-- discrete derivative approximation masks for computing discrete approximations
     of Gaussian derivatives and Gaussian derivative layers in PyTorch.

References:

Lindeberg (1990) "Scale-space for discrete signals", IEEE Transactions on
Pattern Analysis and Machine Intelligence, 12(3): 234--254.

Lindeberg (1993a) "Discrete derivative approximations with scale-space properties: 
A basis for low-level feature detection", Journal of Mathematical Imaging and Vision, 
3(4): 349-376.

Lindeberg (1993b) Scale-Space Theory in Computer Vision, Springer.

Lindeberg (1998) "Feature detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 77-116.

Lindeberg (1998) "Edge detection and ridge detection with automatic scale selection", 
International Journal of Computer Vision, vol 30(2): 117-154.

Lindeberg (2009) "Scale-space". In: B. Wah (Ed.) Wiley Encyclopedia of Computer 
Science and Engineering, John Wiley & Sons, 2009, s. 2495-2504.

Lindeberg (2022) "Scale-covariant and scale-invariant Gaussian derivative 
networks", Journal of Mathematical Imaging and Vision, 64(3): 223-242.


