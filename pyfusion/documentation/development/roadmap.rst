Pyfusion development roadmap
============================

version 0.5
-----------

* full documentation of existing code
* efficient Dictionary of Arrays (DA) npz storage of >20Million Instances
* big data sets (>20MI) selectively extractable from pytables .h5 files
* fftw3 for Fourier

version 0.6
-----------
* python 2/3 compatible
* local .npz storage implemented in acqusition/base to work on all devices
* signal amplitudes able to be extracted from flucstrucs
* re-implement Heliotron-J interface
* metadata API (access to B_0, heating power)
* clustering interface
* capability for efficient I/O (text file?) while doing (multi-process) pre-processing, and put back into sql asynchronously.
* allow separate configuration files for different devices, etc.
* Switch to SafeConfigParser 
  (with Extended Interpolation in 0.7? - but this works only in python 3)

version 0.7
-----------
* implement W7X interface
* re-implement TJII, W7-AS interfaces
* more of Shaun's clustering

version 1.0
-----------

* full feature compatibility with original version

