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

Examples from W7-X
=================

The following examples use command line keyword arguments, see above for
special consideration such as quotes, avoiding spaces within arguments::


  # Clone the repository, then from the shell
  source pyfusion/run_tutorial

  # You will land in ipython. Try example1.py and others as indicated above

  # run_tutorial sets a few env vars and finds the right directory
  # run_pyfusion is an example of a more specialised starting script, 

  # string (--exe) which has some bdb_utils args inside it (in the quotes)  
  # but will need adpatation to your enviroment

  # Then move out of JSPF_examples to a convenient working directory
  cd ../../..

  # make the shot database (shotDA.pickle) , which allows us to obtain
  # cached utc info from shot and date
  run pyfusion/acquisition/W7X/get_shot_list update=1
 
  # run a simple plot script - these data are compact, no need for caching
  run pyfusion/examples/plot_signals.py  dev_name='W7X' diag_name='W7X_TotECH'  shot_number=[20160308,40]

  # A pickle file for OP1.1 (acquisition/W7-X/shotDA.pickle) is included
  # if the 'update' above doesn't work (there are still some unicode/py3 issues).

  # If this does work (or not!) you could try reading and plotting a
  # (trivially short) file of processed data - assumed to be in the top directory
  # (containing pyfusion and .git folders)

  from pyfusion.data.DA_datamining import DA, Masked_DA
  da = DA('LP20160309_52_L57_2k2short.npz')
  da.plot('ne18')  # all channels (with dubious data masked out with Nan's
  da.plot('ne18',select=[1,6]) # selected - LP02 and LP07
  da.keys()  # see data available
  plot(da.masked['Te']) # many dubiuos values masked out
  plot(da['Te'])  # all, including dubious values

  # processing Langmuir data:
  # this should be done with cached data, otherwise the URL access is
  # too slow to be interesting.  Some smaller cached files are included
  # These only have the first few ms of data - see ... for larger files.
  from pyfusion.data.process_swept_Langmuir import Langmuir_data, fixup
  LP30952 = Langmuir_data([20160309,52], 'W7X_L53_LP0107','W7X_L5UALL')
  LP30952.process_swept_Langmuir(threshchan=0,t_comp=[0.85,0.86],filename='*2k2short')
  # the following is a temporary fix, which doesn't work with the
  # small examples - need a full sized data set - almost all 20
  # probes, and the full time interval
  run pyfusion/examples/plot_LP2D LP20160309_52_L53_2k2.npz
  fixup(da,locals(),suppress_ne=suppress_ne) # set t=0 to ECH start, mask out ne for damaged probes

  # Save to local cache
  run pyfusion/examples/save_to_local.py  dev_name='W7X' diag_name="['W7X_L53_LP0107']" shot_list="[[20160309,52]]"
  # make a much smaller version - limited time, every 4th sample 
  run pyfusion/examples/save_to_local.py  dev_name='W7X' diag_name="['W7X_L53_LP0107']" shot_list="[[20160309,52]]" local_dir=/tmp time_range=[0.85,1.45] downsample=4
  # A range of shots - note quotes
  run pyfusion/examples/save_to_local.py "shot_list=shot_range([20160309,6],[20160309,9])" diag_name='["W7X_TotECH","W7X_L57_LPALL"]'
  # add exception=() to the arg list to stop on error (for debugging)

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

  From version 0.60 time units in general are seconds, this can be changed for plots
  only in the configuration file.
  
  MP -  number of processors (only for scripts that have MP in the
  name) 1 avoids multiprocessing

  overlap - fraction of time segment added to the data  (half before, and half after)
  info - 0 1, 2, 3 retains progressively more descriptive and
  diagnostic text

Feature extraction on Heliotron-J
---------------------------------

1a/ for one shot (or a few)::

  # the MP array has just four members - so this is a quick test. -  see below for use of exception
  run  pyfusion/examples/gen_fs_bands.py n_samples=None df=2e3 exception=() max_bands=1 dev_name="HeliotronJ" 'time_range="default"' seg_dt=1e-3 overlap=2.5  diag_name='HeliotronJ_MP_array' shot_range=[60573] info=0 outfile='PF2_151119_60573'

1b/  for many shots - multi processing::

  # include all the MP and PMPs here - takes a few minutes. 
  # exception=() will stop on any error for debugging.  To skip over
  # errors, and continue processing other good data, use exception=Exception
  # This example uses argparse arguments (e.g. ==MP=3) and one long
  # string (--exe) which has some bdb_utils args inside it (in the quotes)
  run  pyfusion/examples/prepfs_range_mp.py . --MP=3  --exe='gen_fs_bands.py n_samples=None df=2e3 exception=() max_bands=1 dev_name="HeliotronJ" ' --shot_range=range(60619,60550,-1) --time_range='"default"' --seg_dt=1e-3 --overlap=2.5  --diag_name='HeliotronJ_ALL'

Result is a text file(s), which will be merged with others in step 2, to form a
DA (Dictionary of Arrays) object

2/ Merge feature files::

 run pyfusion/examples/merge_text_pyfusion.py  "file_list=glob('PF2_151119*_60*')"

The file 'wildcard' expression above needs to be adjusted to include
the files you generated in step 1a or 1b.  The example given works for 1a/.
For 1b, the file names will depend on the date - e.g. 'PF2_1602*' gets
all output generated in feb 2016 regardless of shot number.

Result is a DA data set

3/ Add other data (B_0, NBI, etc)::

 run -i pyfusion/examples/merge_basic_HJ_diagnostics.py diags=diag_extra+diag_basic dd exception=None
 from pyfusion.data.DA_datamining import DA
 DAHJ60573 = DA(dd)                   #  for 1b, it would make sense to call it DAHJ60k = DA(dd)
 DAHJ60573.save('DAHJ60573.npz')

Result: DA data set including other data for each time segment.

4/ Compare with spectrogram/sonogram::

 run pyfusion/examples/plot_specgram.py dev_name='HeliotronJ' shot_number=60573 diag_name=HeliotronJ_MP_array hold=1
 from pyfusion.visual import window_manager, sp, tog 
 sp(DAHJ60573.da, 't_mid', 'freq', 'amp', 'a12', hold=1) 
  
5/ Clustering::

 # DAHJ60k.npz is already prepared in the hj-mhd DA file area (defined in pyfusion.cfg)
 run pyfusion/examples/cluster_DA.py DAfilename='$DAPATH/DAHJ60k.npz'
 co.plot_clusters_phase_lines()  # show clusters
 # Clusters 0,2,5 look interesting, but the phase difference at 2 in
 #   all these looks out by pi - is this a wiring change?

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
