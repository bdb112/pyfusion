Examples from the JSPF tutorial article
========================================
These are the simplest examples  - no command line args except -i
when you want the data preserved (see example1a, 2 etc) 

.. literalinclude:: ../../examples/JSPF_tutorial/example1.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example1a.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example2.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example2a.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example3.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example4.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example5.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example6.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/cross_phase.py
   :caption:


.. _advanced-examples:

More advanced examples
========================

These examples use arguments on the command line to adjust the
behaviour - e.g. NFFT=1024

In most cases, arguments are parsed by a routine (bdb_utils) in this package,
with the following properties

   * there should be no spaces around the equals sign, NFFT=2048 not NFFT = 2048

   * quotes may be needed around parentheses - 

   * typical arguments and defaults can be seen by using the keyword help

   * the keyword exception may be used to skip over exceptions (e.g. bad shot data)
        exception=Exception          
     or to stop ready to enter the debugger (%debug) on an exception
        exception=()
     Note the case distinction (Exception is a researved word,
     exception is not)

Example from Heliotron-J in four steps
======================================================
There are a lot of parameters here, but it is only necessary to
adjust the shot_range, diag_name and the outfile name (and possibly the value of
seg_dt, the time interval in ms).  Once you have examined the data
produced in some detail, then you may want to adjust the other
parameters.

  max_bands - increase if there are more than a few simultaneous frequencies

  time_range - automatically finds MHD activity on HeliotronJ at
  least - but you can restrict the time range by setting this

  MP -  number of processors (only for scripts that have MP in the
  name) 1 avoids multiprocessing

  overlap - fraction of time segment added to the data  (half before, and half after)
  info - 0 1, 2, 3 retains progressively more descriptive text

Feature extraction on Heliotron-J
---------------------------------
1a/ for one shot (or a few)::

  # the MP array has just four members - so this is a quick test. -  see below for use of exception
  run  pyfusion/examples/gen_fs_bands.py n_samples=None df=2. exception=() max_bands=1 dev_name="HeliotronJ" 'time_range="default"' seg_dt=1. overlap=2.5  diag_name='HeliotronJ_MP_array' shot_range=[60573] info=0 outfile='PF2_151119_60573'

1b/  for many shots - multi processing::

  # include all the MP and PMPs here - takes a few minutes. 
  # exception=() will stop on any error for debugging.  To skip over  errors, use exception=Exception
  # This example uses argparse arguments (e.g. ==MP=3) and one long
  # string (--exe) which has some bdb_utils args inside it (in the quotes)
  run  pyfusion/examples/prepfs_range_mp.py . --MP=3  --exe='gen_fs_bands.py n_samples=None df=2. exception=() max_bands=1 dev_name="HeliotronJ" ' --shot_range=range(60619,60550,-1) --time_range='"default"' --seg_dt=1. --overlap=2.5  --diag_name='HeliotronJ_ALL'

Result is a text file(s), which is then merged with others, to form a
DA (Dictionary of Arrays) object

2/ Merge feature files::

 run pyfusion/examples/merge_text_pyfusion.py  "file_list=glob('PF2_151120*_60*')"

Result is a DA data set

3/ Add other data (B_0, NBI, etc)::

 run -i pyfusion/examples/merge_basic_HJ_diagnostics.py diags=diag_extra+diag_basic dd exception=None
 DAHJ60k = DA(dd)
 DAHJ60k.save('DAHJ60k.npz')

Result: DA data set including other data for each time segment.

4/ Clustering::

 # DAHJ60k.npz is already prepared in the hj-mhd DA file area (defined in pyfusion.cfg)
 run pyfusion/examples/cluster_DA.py DAfilename='$DAPATH/DAHJ60k.npz'
 co.plot_clusters_phase_lines()  # show clusters
 # Clusters 0,2,5 look interesting, but the phase difference at 2 in
 #   all these looks out by pi

 # alternatively, check the one you just prepared in the previous step
 run pyfusion/examples/cluster_DA.py DAfilename='DAHJ60k.npz'

Setting up hj-mhd
=================

operating system::

 install f77 and tig  # (git utility)
 # assuming git etc already there

the rest is there already or in anaconda.

anaconda:
Best way is to download both python2 and 3 download files, and keep them for later.
Easier to setup when the /homea drives are present (booted the other way).

Install anaconda as on the web::

 bash...

Notes
- better to install in the folder they recommend - ~/anaconda2  etc.
This wastes a little space, but is simpler.  It is not that easy to move (your username etc gets written into at least 1000 different places)- much faster to just reinstall.