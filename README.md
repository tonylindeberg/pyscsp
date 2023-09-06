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
    (blob detection or corner detection) and ridge detection
	
(see more detailed comments in the source code for additional information)


* torchscsp.py: Subset of functionalities for use in PyTorch:

-- functions for generating 1-D discrete approximations of the Gaussian kernel
     for spatial smoothing with separable filtering in PyTorch
	 
-- discrete derivative approximation masks for computing discrete approximations
     of Gaussian derivatives in PyTorch

