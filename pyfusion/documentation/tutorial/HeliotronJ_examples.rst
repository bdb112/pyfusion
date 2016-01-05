Examples from the JSPF tutorial article
========================================
.. literalinclude:: ../../examples/JSPF_tutorial/example1.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example1a.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example2.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example3.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example4.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example5.py
   :caption:

.. literalinclude:: ../../examples/JSPF_tutorial/example6.py
   :caption:

A more advanced example in four steps
=======================================

Feature extraction on Heliotron-J
---------------------------------
1a/ for one shot (or a few)::

  run  pyfusion/examples/gen_fs_bands.py n_samples=None df=2. exception=None max_bands=1 dev_name="HeliotronJ" 'time_range="default"' seg_dt=1. overlap=2.5  diag_name='HeliotronJ_MP_array' shot_range=[60573] info=0 outfile='PF2_151119_60573'

1b/  for many shots - multi processing::

 run  pyfusion/examples/prepfs_range_mp.py . --MP=3  --exe='gen_fs_bands.py n_samples=None df=2. exception=None max_bands=1 dev_name="HeliotronJ" ' --shot_range=range(60619,60550,-1) --time_range=\'\"default\"\' --seg_dt=1. --overlap=2.5  --diag_name=\'HeliotronJ_ALL\'

Result is a text file(s), which is then merged with others, to form a
DA (Dictionary of Arrays) object

2/ Merge feature files::

 run pyfusion/examples/merge_text_pyfusion.py  "file_list=glob('PF2_151120*_60*')"  target="b'Shot .*"

Result is a DA data set

3/ Add other data (B_0, NBI, etc)::

 run -i pyfusion/examples/merge_basic_HJ_diagnostics.py diags=diag_extra+diag_basic dd exception=None
 DAHJ60k = DA(dd)
 DAHJ60k.save('DAHJ60k.npz')

Result: DA data set including other data for each time segment.

4/ Clustering::
 run pyfusion/examples/cluster_DA.py DAfilename='DAHJ60k.npz'
 co.plot_clusters_phase_lines()  # show clusters

  

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
