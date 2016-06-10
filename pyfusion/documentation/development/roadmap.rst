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

version 0.7
-----------
* implement W7X interface including gas, currents, ECE 
* fix coordinate code to work with W7X, both direct and cached data
* metadata API (access to B_0, heating power)
* error estimation for LP data
* Correlation for probe data with other signals (so far only in a script)
* save to CSV, JSON
* generic mini summary

version 0.8
-----------
* clustering interface
* capability for efficient I/O (text file?) while doing (multi-process) pre-processing, and put back into sql asynchronously.
* allow separate configuration files for different devices, etc.
* Switch to SafeConfigParser 
  (with Extended Interpolation in 0.7? - but this works only in python 3)



version 0.9
-----------
* re-implement TJII, W7-AS interfaces
* more of Shaun's clustering

version 1.0
-----------

* full feature compatibility with original version

