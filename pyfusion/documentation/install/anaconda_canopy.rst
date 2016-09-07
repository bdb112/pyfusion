
##############################################
Installing pyfusion using precompiled packages
##############################################

:Release: |version|
:Date: |today|

.. _install-anaconda:
Anaconda
========

Required
--------
 * perform base installation, using bash in your home directory 
 * conda install scikit-learn

Recommended 
-----------
 * git (e.g sudo aptitude install git-core) allows accessing other 
   versions of pyfusion, and is recommended to use for further development.
 * tig or alternative to browse the git repository
 * f77 for access to Heliotron-J data
 * MDSplus for access to H-1 data (beyond the small extract in the
   download package)
 * retrieve, retrieve_t, igetfile for access to LHD data (contact LHD data group)
 * cython for some probability density clustering routines
   (sudo aptitude install cython)
 * fftw3 to speed up some pyfusion Fourier transforms 
   (sudo aptitude install fftw3), and python interface  - (easy_install pyfftw)

.. _install-canopy:
Canopy
======

Required
--------
 * base installation, e.g. bash canopy-1.5.3-rh5-64.sh (presently
   known as express)
 * conda install scikit_learn using Package manager
 * Use canopy terminal for the examples, starting with source
   pyfusion/run_tutorial as described elsewhere.

See `Recommended` above

Installing pyfusion
===================

Old instructions follow - need updating.
---------------------------------

At present, the recommended method of installing pyfusion is from the
code repository. There is no need to run setup.py.  The small number
of non-python files (mainly for specific fusion device libraries)
are meant to compile 'on the fly'.

You'll need a directory in your PYTHONPATH to install to, eg::
   
   mkdir -p $HOME/code/python
   echo "export PYTHONPATH=\$PYTHONPATH:\$HOME/code/python" >> $HOME/.bashrc
   source $HOME/.bashrc

Install the `git <http://git-scm.com/>`_ distributed version control system::

	sudo apt-get install git-core

Make a clone of the pyfusion repository in your python path::

     cd $HOME/code/python
     git clone https://github.com/bdb112/pyfusion
     # obsolete version http://github.com/dpretty/pyfusion.git

Until version 1.0 of the code, we'll be using the dev branch, so you need to check it out::

     cd pyfusion
     git checkout -b dev origin/dev
 
