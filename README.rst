
pyfusion - python code for data mining plasma fluctuations
----------------------------------------------------------

Most recent update: 

***  Bug! 20160824  save of a file already in local cache (e.g. to decimate) seems to square gains.***
 * partial fix - don't allow local_saves from local cache (only works
   on machines with acccess to the archivedB
 maybe use functools.wraps so that the __doc__ of plot_signals can be seen

Version 0.7.8.beta

* include raw dimension utcs in data.params - can be used to try to recontruct bad time vectors.
* save_compress py3, save_to_local - save logs as json.
* document valid_dates
* many impronements to W7X_neTe_profile, cmd_line, Pdsmooth, median, compensation, profile fits
* also plot_both_LP2D.py
* mini_summary includes text and MDS version
* pyfusion.cfg - add more valid_dates, and add individual ECH chans
* W7X_read_json - for testing url reads off line

Version 0.7.7 alpha

* Add a valid_dates feature to base.py so that pyfusion.cfg can have
  changes to parameters for specific date ranges.
* implement for L53_LP05-12 - need to do converse for LP_U
* Also simple check that params['DMD'] is consistent between npz.file
  and pyfusion.cfg
* add no_cache option to getdata so that the local cache can be
  avoided, (activate by save_compress=0 in save to_local for now)


Version 0.7.6 alpha

* change W7X shot to a tuple (reason for calling an alpha)
* debug some error messages in W7X
* fix images in README.rst
* make the feedback about which shotDA file is used only print for VERBOSE>0
* fix units and magnitude error in puff_db
* integrate filter function had a confused baseline removal - now fixed and allows for constant and slope removal
* added hold=2 option to plot_signals.py to put such data on a second y axis 
  (also in data/plots allow plotting a single channel on an existing axis for overplotting etc)
* converted mini_summary to use pure pyfusion
* improvements to plot_both_LP2D, debug weighted averaging
* get_shot_list - info messages suppressed unless VREBOSE>0
* acq/data/base - keep track of data source (source via acq.source) in params 
* several Langmuir file - change Vp to Vf
* N2_puff_correlation - move ECH to a twin axis, imporve limit
  setting
* extract_limiter_coords - extract limiter profile in midplane, include node index list

See below for previous updates


Pyfusion code
-------------

This is my fork of David Pretty's original pyfusion code, with much
input recently from Shaun Haskey. The code runs in 2.6+ and most of the
code is compatible with python
3.3+.(https://github.com/bdb112/pyfusion). The 'six' module is required
for both pythons for commits >= fb757c75

For python 2, release >205b21 is recommended use with the tutorial
article in JSPF 2015, although all later releases should also work. The
latest release is recommended for python 3.

JSPF tutorial
-------------

A tutorial article will appear soon in
http://www.jspf.or.jp/eng/jpfr\_contents.html (in Japanese) and will be
posted on the H-1 heliac website in english, along with full
documentation of pyfusion (now at
http://people.physics.anu.edu.au/~bdb112/pyfusion/). In time, the latest
docs will be automatically generated on readthedocs.org.

To run the examples therein, install the files from the zip or the git
repository anywhere, and do

.. raw:: html

   <pre><code>
   source pyfusion/run_tutorial     # or wherever you installed it
   </code></pre>

This will add the pyfusion path to your PYTHONPATH, and cd to the
JSPF\_tutorial directory, and put you into ipython. Then try

.. raw:: html

   <pre><code>
   In [1]: run example4.py
   </code></pre>


Quick Installation
------------------

Install the default anaconda or canopy python environment for python 3.
For anaconda, add

.. raw:: html

   <pre><code>
   conda install scikit-learn
   </code></pre>

For more details see

.. raw:: html

   <pre><code>
   http://people.physics.anu.edu.au/~bdb112/pyfusion/tutorial/install/index.html
   </code></pre>


Extract from the Tutorial Article "Datamining Applications in Plasma Physics"
-----------------------------------------------------------------------------

High temperature plasma has many sources of magnetic and kinetic energy,
which can drive instabilities. These may disrupt the plasma, damage
components in the plasma vessel, or at best waste energy, reducing
efficiency. Achieving efficient, economic fusion power requires that
these instabilities be understood, and with this knowledge, controlled
or suppressed.

**What are the objectives?**:

1. Identify the physical nature of plasma modes - oscillations or fluctuations
2. Distill large data sets describing these into a data base of a manageable size.
3. With this knowledge, develop means of automatically classifying and identifying these modes.

Datamining helps with all these aims, especially in automating the process.  This enables the use of large datasets from the entire operational life of many plasma confinement devices, well beyond the capability of analysis by hand.  Ultimately this will enable near real-time identification of modes for control and feedback.

**What are the modes of interest?**:
By plasma modes we mean plasma oscillations which will usually be incoherent to some extent , because plasma parameters such as density vary in time and in space.  If we can measure the frequency, and its dependence on plasma parameters, we can have some idea of the plasma wave associated with it.  It is better still if we can learn something about the wavelength, or more generally the k vector, so we can in essence measure a point on the dispersion relation of the underlying wave.  Typical modes are drift wave oscillations and Alfvén instabilities. Modes may be driven for example by ideal or resistive MHD instabilities, or by transfer of energy from fast particles, especially if the particle velocity is related to the wave velocity such that a resonant interaction occurs.  The extraction of wavelength information implies the existence of more than one channel of data, so this paper is focussed on analysis of multi-channel time-series data.  

**Installation notes**:
Note that the "source" command is used above because it is necessary to set some environment variables, and simply running a script will not - any environemnt changes are discarded.  Also, although these examples work with straight python, ipython is recommended because of the ease of inspectin variable, debugging, and recalling history.  Features include the use of ? for help informatin and tabbing to see possible completions.  More advanved features can be enabled by settings in ~/ipython/profile_default/ipython_config.py, such as automatically supplying parentheses, automatically reloading imported modules if they are edited.

In the spirit of the version control package 'git', the user is encouraged to work in the source directory structure.  If git is used, the source files are safe, and you can easily see the changes you have made.  This requires that the user has write permission ford this directory, which happens by default if you clone the repository.  

.. raw:: html

   <pre><code>
   git clone /home/bdb112/pyfusion/mon121210/pyfusion/
   cd pyfusion
   </code></pre>

If you don't have write permission, many of the examples will not complete.  <code>git diff </code> will show your changes, but if you want to run previous versions, casual users of git should note that <code>git checkout </code> will silently overwrite any changes you have made to files that came from the repository, so you should use <code>git stash </code> to save your current work, or make another clone.

Example output
--------------

| Example clustering showing Alfvenic scaling in the H-1 heliac.
|

.. image:: pyfusion/6_good_clusters_CPC.png

| Example of mode identification in the LHD Heliotron at the National Institute of Fusion Science, Toki.
| 

.. image:: pyfusion/65139_N_mode_id_new.png


**Relevant publications include:**:

1. D. G. Pretty and B. D. Blackwell.   Comp. Phys. Comm., 2009. http://dx.doi.org/10.1016/j.cpc.2009.05.003 and thesis 
2. SR Haskey, BD Blackwell, DG Pretty, Comp. Phys. Comm. 185 (6), 1669-1680, http://dx.doi.org/10.1016/j.cpc.2014.03.008 and thesis


Previous Updates
----------------

Version 0.7.5 beta 

* integrate doc and update README.rst, eliminate README.md
* get_shot_list - nicer output format
* data/base.py warn if cached data is in a temp dir
* DA_info optional 3rd positional argument - key to examine
* process_Langmuir - rearrange so that mask can be re set by simple paste
* mini_summary - add some more diags
* plot_both_LP2D.py - plot upper and lower segs together, only some
  improvements back ported to plot_LP2D.py
* run_process_LP - changed tcomp to slightly smaller to allow for
  early breakdown
* partial fix of save_to_local - don't allow local_saves from local cache (only works
  on machines with acccess to the archivedB
* save_to_local saves log in a pickle

Version 0.7.4 beta

* delayed MDSplus import to avoid import error for JSPS example1
* several small improvements, incl minpts arg to plot_LP2D.py, generalise run_process_LP,
* tune tests to make more test_examples work, failed attempt to implement timeout in test-examples
* add branch lukas

Version 0.7.3 alpha

* comment fields now included and recognised in pyfusion.cfg files
* pyfusion.cfg space chars in URLS changed from %20 to %%20 for py3
* W7X examples added, including some very short data files for practice/debug
* Add Ie/Ii ratio to dataset (Ie_Ii)
* Adapt DA_datamining to use on h1
* fix bug in mdsplus style paths
* explore alternative corrections to corrupted timebase - but leave suppressed
* centralise access to shotDA.pickle/json
* Test routine (test_examples.py) now only tries file in the git
  repo, optionally newest first
* edit several new example routines to run under test.
* replace inf in JSON write_LP_as_CSV.py some matlab doesn't
  recognize inf?  loadjson.m (mathworks, qianqian fang 2011/09/09
  seems to want to read Inf.


Version 0.7.2: beta

* minor fixes to get working on H-1 data again (shot, config_name,
     config_boyd) implement averaging through lists in plotLP2D


Version 0.7.1: beta

* make 't' the default time variable (if t_zero is given) in Langmuir
  DA files.  't' is derived from t_mid:  t = t_mid - t_zero
* filters.py: now segment() accepts floats for the number of samples,
  allowing the segments to be phase locked to a signal.
* process_swept_Langmuir also.
* N2 puff correlation - generalise and tidy, labelling
* W7X_neTe_profile - fix sign error in 'x' coord
* plot_LP2D  - add acquisition/W7X/puff_db, suppress dodgy ne in
  image, get seg 7 axes right way up.


Version 0.7.0: beta

* process_swept Langmuir 
  threshchan is used to determine start and end of plasma
  residual DC offset removed in get_iprobe
  IO too small used in mask criterion
* plot_LP2D - general improvements
* N2_puff_correlation - choice of physical units or coefficient
* write_LP_as_CSV also writes JSON

Version 0.7.0: alpha

* fixes to leastsq, add error estimates through covariance (leastsq only) and by
  tracking the convergence in time (both amoeba and leastsq)
* also fit has LP filter option and removal of unrelated harmonics
* LP_extra has pre-fit filtering and error estimation
* get_LP_data improvements, filtering etc.
* add hist() function to DA_datamining

Version 0.6.9: beta

* temporary update to avoid too many changes at once
* process_swept_Langmuir includes scipy.optimise.leastsq and some more
  parameters, also tracks the root finder, fixed figure count limiter
  and imporved flexibility of saved filename.
* get_LP_data.py pulls the v,i data from a characteristic plot and
  plays with it for algorithm development.
* N2_puff_correlation: add correlation  (coefficient and physical
  units) and Lukas's distance routine.
* add write_LP_as_CSV.py (also JSON)
* add examples/correct_LP_data.py, file_sorter and file_finder_db which allows
  local_data cache to be rationalised
* fix bug in save_compress brought on by corrupted W7X timebase
* Raise Error if data is pre 0.68b
* improve auto filename generation in process_swept_Langmuir
* pyfusion.cfg corrections (delete LP21..)
* fix domain checker to retain result in self.acq
* add mdsplus style path extra_data/to organise shots into folders 
* calc correlation in examples/N2_puff_correlation.py, also lukas probe info including distance to LCFS
* minor fixes to fourier in data/filters.py
* mini_summary.py try speeding up sqlite file form, make less MDSplus dependent
* improvements for plot_LP2D.py

Version 0.6.8: beta

* Corrected limiter swap (3 and 7 interchanged) and several typos.
* Added time plot of diagnostics to plot_LP2D.py
* moved dummysig into data.filters module

Version 0.6.7: alpha

* Corrected Langmuir probe coordinates 11-20, added areas from Tipflachen_boyd.xlsx, added host and incremented npz version to 103 to indicate correct coords.
* examples/N2_puff_correlation.py uses ECH start as time zero
* Add gas controllers, currents including MainCoils, TrimCoils
* Many improvements to process_swept_Langmuir, including actual_params
* Simple test to warn if process is unable to access ipp-hgw, to avoid
  waiting for timeout accessing URL

Version 0.6.6:
 
* restores coordinates coding (incl W7X), transforms not properly implemented yet
* process_swept_Langmuir is more convenient to use (incl auto load and save)
  rest_swp='auto' choose to restore the sweep according to shot number.
* plot_LP2D - animate Te and ne (into pngs)  
* examples/modify_cfg.py is a script to add/modify pyfusion.cfg
  (presently coordinates)
* Some gas controls in pyfusion.cfg
* pyfusion.DBG() instead of pyfusion.DEBUG if a purely numeric value is
  needed (e.g. in > or < tests).  This avoids unwanted debugger breaks when a
  text key is used.
* fixed problem in LHD data access due to exception in LHDConvenience function.  (output_coords)

Version 0.65: Langmuir processsing is separated into two classes/objects (see
data/process_swept_Langmuir), optimised and saved as dictionary of
array (DA) files, with a built in mask of dubious data.
Clipped sweep voltage can be restored by restore_sin()

Issues: 

1. applying restore_sin to data that are not clipped produces
   large errors.
2. partial clipping produces elevated Te
3. fit quality criterion and ne calculation need improvement

**Version 0.64** beta has improved processing of clipped, swept Langmuir probe data,
Next version will have multi-channel data extraction system using pyfusion 'Dictionary of Arrays'.

**Version 0.63 beta** has fixes for multichannel diagnostic local saves,
and convenient entry for large ranges of data and shots.
Initial Langmuir analysis in process_swept_Langmuir, and pyfusion.CACHE to
allow local chaching of json data. (very large!)

pyfusion.reload() to reload configuration - git 5aed of 3-Mar

Version 0.62 alpha includes more timebase checks for W7X, corrected
gains for channels, and saves utc and params with data.

**Version 0.61** includes first working version of W7-X archiveDB
support, without much care for python3 compability of the new code.
beginning support for two component shot number e.g. [20160301,5]

**Best pre W7X Version (0.60) is 09ba5** - supports Python 2/3 for almost all scripts 
(MDSplus is the main problem - see issues) and the full set of examples in the JSPF tutorial article. 
The 4 criteria on the development roadmap have been achieved, and the
five that were postponed until 0.7 are at least partially
implemented.

**Version 0.58** now supports the full set of examples in the JSPF
tutorial article, and includes the data files (in downsampled form). All
will run in the download package, apart from two marked (\*) requiring
access to full databases.

.. raw:: html

   <pre>
   example1.py
   example1a.py
   example1_LHD.py*
   example2.py*
   example3.py
   example4.py
   example5.py
   example6.py
   </pre>

