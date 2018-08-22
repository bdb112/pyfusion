.. _config-files:

Configuration files
"""""""""""""""""""

Overview
--------

Pyfusion uses simple text files to store information such as data acquisition settings, diagnostic coordinates, SQL database configurations, etc. A pyfusion configuration file looks something like this::

 # Some comments are indented below to show the actual options better 
 # Indentation of comments is only allowed in version 0.6 onwards

 [global]
   # this section is a pyfusion feature to hold values that are universal
   # These values have to be specifically queried.
 database = sqlite:///:memory:
 tmpdir = /tmp

 [DEFAULT]
   # This section is a standard configparser feature, and supplies
   #  options to all sections.  The main use is to define substitutions
 H1fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
   # The presence of these options or "keys" in all other sections can be
   # annoying, and future versions may use ExtendedInterpolation to
   # separate interpolations, thereby reducing key 'pollution'.
 
 [Device:H1]
 dev_class = pyfusion.devices.H1.device.H1
 acq_name = MDS_h1
 
 [Acquisition:MDS_h1]
 acq_class = pyfusion.acquisition.MDSPlus.acq.MDSPlusAcquisition
 server = h1data.anu.edu.au
 
 [CoordTransform:H1_mirnov]
 magnetic = pyfusion.devices.H1.coords.MirnovKhMagneticCoordTransform
 
 [Diagnostic:H1_mirnov_array_1_coil_1]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_1
 coords_cylindrical = 1.114, 0.7732, 0.355
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1_coil_2]
   # example using interpolation or substitution instead of literal
 data_fetcher = $(H1_fetcher)s
 mds_path = \h1data::top.operations.mirnov:a14_14:input_2
 coords_cylindrical = 1.185, 0.7732, 0.289
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1_coil_3]
 data_fetcher = pyfusion.acquisition.H1.fetch.H1DataFetcher
 mds_path = \h1data::top.operations.mirnov:a14_14:input_3
 coords_cylindrical = 1.216, 0.7732, 0.227
 coord_transform = H1_mirnov
 
 [Diagnostic:H1_mirnov_array_1]
 data_fetcher = pyfusion.acquisition.base.MultiChannelFetcher
 channel_1 = H1_mirnov_array_1_coil_1
 channel_2 = H1_mirnov_array_1_coil_2
   # This next line corrects a phase flip error 
   # better to do this in the multi-channel diag - more flexible and
   # works well with local data in *npz files.
 channel_3 = -H1_mirnov_array_1_coil_3



There are two types of sections in this file: there are two `special`
sections (global, DEFAULT) and several `component` sections (e.g. Device:H1, Acquisition:MDS_h1, CoordTransform:H1_mirnov, etc.)

See :ref:`configparser-basics` which includes some syntax rules.

  .. ********** EDIT LINE. Is this where Dave got up to ??  ***********



The sections in the configuration (except for [variabletypes]) file have the syntax
[Component:name], where Component is one of: Acquisition, Device,
Diagnostic. When instantiating a class, such as Device, Acquisition,
Diagnostic, etc. which looks in the configuration file for settings,
individual settings can be overridden using the corresponding keyword
arguments. For example, ``Device('my_device')`` will use settings in
the ``[Device:my_device]`` configuration section, and
``Device('my_device', database='sqlite://')`` will override the
database configuration setting with ``sqlite://`` (a temporary in-memory database).  


The pyfusion configuration parser :class:`pyfusion.conf.PyfusionConfigParser` is a simple subclass of the `standard
python configparser
<http://docs.python.org/library/configparser.html>`_, for example, to
see the configuration sections, type::

    pyfusion.config.sections()

Valid Shots
-----------
A new feature allows configuration to change for different shot
ranges.  Initially the shots work back from the latest config.  If for
a particular diagnostic, the shot is outside the valid_shots, then
alternate diag names such as W7XM1_L53_LP02_I are checked for in the
config file.  If found, and the shot range matches, we are finished. 
Otherwise an error is generated.

A second modification (M2) builds on the first (M1), so the effect is
cumulative.  If a diagnostic is missing on a day, it will have to be
restored on the previous day. Diagnostics can be suppressed for now by
setting DMD=0, so all the other charactersitics remain, so it can be
easily restored.


1/ If a single channel diagnostic has no 'valid_shots', the value of its .acq valid_shots is used.

2/ If a shot is requested outside the valid range, a series of modifier diagnostic entries Mn, where n is an integer are searched for until one with a valid_shot range which includes that shot.
 
An extract from pyfusion.cfg as an example::

  [Acquisition:W7X]
  acq_class = pyfusion.acquisition.W7X.acq.W7XAcquisition
  domain = ipp-hgw.mpg.de
  # Look for this string in the list of name servers for domain - if not found,
  #   then the URL is probably not accessible through this connection.
  # Avoids an unnecessary (long) wait to see error messages.  
  lookfor = sv
  valid_shots = L53_LP_from=20160122,L53_LP_to=20160310,L57_LP_from=20160122,L57_LP_to=20160310

  ----- further down in the file ----

  [Diagnostic:W7X_L53_LP05_I]
  valid_shots = L53_LP_from=20160223,L53_LP_to=20160310
  coords_w7_x_koord = 1.72390, -5.41380, 0.21680
  area = 0.963e-06
  sweepv = W7X_L53_LP01_U
  gain =  %(1Ohm)s
  data_fetcher = %(W7Xfetcher)s
  params = CDS=82,DMD=190,ch=0

  [Diagnostic:W7XM1_L53_LP05_I]
  #The first modification says for the shots from 1/22 to 2/18, use a different DMD (digitiser box address)
  valid_shotss = L53_LP_from=20160122,L53_LP_to=20160218
  params = CDS=82,DMD=184,ch=0

So in this example shots on 20160224 would get the parameters from the
main entry (W7X_L53_LP05_I) but a shot on the 18th Feb or earlier would get the main entry, but with the new params line, including DMD=184

The effect is cumulative, so there is no need to repeat the unchanged parameters.

The LHS of the shot (e.g. L53_LP) restricts the application to
diagnostics matching the letters before '_from' and '_to' .  This is
pretty crude, but is needed so that inheritance form .acq will work.
Otherwise there has to be a valid_shot in every entry.   (which may be
a good thing in the long term, but to much work for now. - only the
diagnostics that change need valid shots for now.) Another reason for
a 'selective' shot range, one that applies to a select range of
diagnostics, is that %() substitution can be used to simplfy edits.

This 'working backwards' seems a natural fit to the way the changes were made, and fits well with Soren's excel, but it would need modification if a new configuration was used at a later date.  Some M1's would need to be edited into M2's etc.  Messy.  I can't easily see how to make a scheme working forward in time, but we don't need that until W7-X Op1.2!  


Loading config files
--------------------
When pyfusion is imported, it will load the default configuration file
provided in the source code (that is in the pyfusion directory)
followed by your custom configuration file, 
in ``$HOME/.pyfusion/pyfusion.cfg``, if it exists. 
and finally files pointed to by the environment variable PYFUSION_CONFIG_FILE
if they exist. This allows temporarily overriding config variables.

The user's own custom file ``$HOME/.pyfusion/pyfusion.cfg`` contains information specific to the machine it is on and the user::

 # this is an example of .pyfusion/pyfusion.cfg - the user's personal config -
 #   ~/.pyfusion/pyfusion.cfg on linux
 #   c:/Users/bobl/.pyfusion/pyfusion.cfg on window including cygwin
 [global]
 localdatapath = C:\cygwin\home\bobl\data\datamining\local_data\+\\sv-w7x-nas-1\Freigaben\bobl\LLPcache\W7X\~d~c~b~a
 # other examples - windows cygwin
 #localdatapath = C:\cygwin\home\bobl\data\datamining\local_data\+C:\cygwin\home\bobl\pyfusion\working\pyfusion\may2016\~d~c~b~a
 # linux
 #localdatapath=/data/+/data/datamining/local_data/extra_data/may22/0218/+/data/datamining/local_data/extra_data/may2016/~d~c~b~a
 #localdatapath = .




Additional config files can be loaded with ``pyfusion.read_config()``::

	   pyfusion.read_config(["another_config_filename_1", "another_config_filename_2"])

The ``read_config`` argument can either be a single file-like object
(any object which has a ``readlines()`` method) or a list of
filenames, as shown above. If you do not supply any argument,
``read_config()`` will load the default configuration files (the same
ones loaded when you import pyfusion). 

To clear the loaded pyfusion configuration, use
``pyfusion.conf.utils.clear_config()``. If you want to return the configuration
to the default settings (the configuration you have when you import
pyfusion), type::

	   pyfusion.conf.utils.clear_config()
	   pyfusion.read_config()

Will include a clear in the reload_config convenience function - but this has been disabled because of problems with the sequence/and/or the wat to access clear.
See :ref:`testing-config`

Using translation from readable 'views/KKS' names to coda channels
--------------------------------------------------------------
Follow the archiveDB links down from views/KKS until the path switches to coda, then try the last non coda link with scaled/ added.

From the address box:
http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/KKS/CDX21_NBI_Box21/DAQ/BE000/?filterstart=1533772800000000000&filterstop=1533859199999999999
The link name is:
HGV_1 Monitor U 
Remove up to KKS/ from the two joined
http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/views/KKS/CDX21_NBI_Box21/DAQ/BE000/HGV_1 Monitor U
get_signal_url("CDX21_NBI_Box21/DAQ/BE000/HGV_1 Monitor U")
/scaled/
Found!
Out[9]: 'http://archive-webapi.ipp-hgw.mpg.de/ArchiveDB/codac/W7X/CoDaStationDesc.31/DataModuleDesc.24119_DATASTREAM/0/HGV_1%20Monitor%20U'
Then access the result + /scaled/ + _signal etc

So the line in pyfusion.cfg is just the result + /scaled/ typically, although the presence of /signal/ will be ignored (removed)

Debugging .cfg files
--------------------
pyfusion.DEBUG=3 is enough to suppress try/excepts so that the actual
error is seen




[variabletypes]
---------------
`[Does not seem to be fully implemented as of Dec 2015 - it appears
only in some test.cfg files.  This is probably because in most cases,
the code knows the type.  Only in Diagnostic: sections does the
information get interpreted by non-specific code (put into a dictionary) ]`.

variabletypes is a section for defining the types (integer, float,
boolean) of variables specified throughout the configuration file. By
default, variables are assumed to be strings (text) - only variables
of type integer, float or boolean should be listed here.

For example, if three variables (arguments) for the Diagnostic class
are n_samples (integer), sample_freq (float) and normalise (boolean)
the syntax is:: 

	Diagnostic__n_samples = int
	Diagnostic__sample_freq = float
	Diagnostic__normalise = bool

Note the double underscore (__) separating the class type and the
variable name.

[Device:name]
-------------

database
~~~~~~~~

Location of database in the `SQLAlchemy database URL syntax`_. 

e.g.::
   
   no example yet

.. _SQLAlchemy database URL syntax: http://www.sqlalchemy.org/docs/04/dbengine.html#dbengine_establishing

acq_name
~~~~~~~~

Name of Acquisition config setting ( [Acquisition:acq_name] ) to be used for this device.

e.g.::

   acq_name = test_fakedata

dev_class
~~~~~~~~~

Name of device class (subclass of pyfusion.devices.base.Device)
to be used for this device. This is called when using the convenience
function pyfusion.getDevice. For example, if the configuration file
contains::

	[Device:my_tjii_device]
	dev_class = pyfusion.devices.TJII

then using::

     import pyfusion
     my_dev = pyfusion.getDevice('my_tjii_device')

``my_dev`` will be an instance of pyfusion.devices.TJII

[Acquisition:name]
------------------

acq_class
~~~~~~~~~

Location of acquisition class (subclass of pyfusion.acquisition.base.BaseAcquisition). 

e.g.::
  
   acq_class = pyfusion.acquisition.fakedata.FakeDataAcquisition

[Diagnostic:name]
-----------------


data_fetcher
~~~~~~~~~~~~

Location of class (subclass of pyfusion.acquisition.base.BaseDataFetcher) to fetch
the data for the diagnostic.

tests.cfg
---------

A separate configuration file "tests.cfg", in the same ".pyfusion" folder in your home directory, can be used during development to enable tests which are disabled by default.

An example of the syntax is::

	[EnabledTests]
	mdsplus = True
	flucstrucs = True


Database
--------
The database layer is handled by `SQLAlchemy <http://www.sqlalchemy.org>`_ 

.. _db-urls:

Database URL
~~~~~~~~~~~~

Database URLs are the same as for SQLAlchemy::

	 driver://username:password@host:port/database

For more details, refer to http://www.sqlalchemy.org/docs/05/dbengine.html#create-engine-url-arguments 

.. _configparser-basics:

Configparser basics
-------------------
Notes:

* python 3 configparser.ConfigParser is more strict than the python2
  ConfigParser.ConfigParser (newer python 2 versions have
  SafeConfigParser which is very close of not the same as python 3 
  ConfigParser.

* pyfusion.config... accesses the standard python configparser functions, such as
  ``pyfusion.config.get('Diagnostic:MP1','DIAG_NAME') --> 'FMD'``
  whereas

* pyfusion.conf. accesses the pyfusion specific functions (see example
  below, note that the section name is given in two parts there
  ('Diagnostic','MP1') 

* Anything in the [DEFAULT] section will appear in the scope of the section (i.e. the
  dictionary returned by ``pyfusion.conf.utils.get_config_as_dict()``

e.g.::

 pyfusion.conf.utils.get_config_as_dict('Diagnostic','MP1')
 {'channel_number': '18',
  'coord_transform': 'LHD_convenience',
  'coords_reduced': '18.0, 0.0, 0.0',
  'data_fetcher': 'pyfusion.acquisition.LHD.fetch.LHDTimeseriesDataFetcher',
  'diag_name': 'FMD',
  'filepath': '/tmp/LHDtmpdata',
  'gain': '1',
  'hjfetcher': 'pyfusion.acquisition.HeliotronJ.fetch.HeliotronJDataFetcher',
  'lhdfetcher': 'pyfusion.acquisition.LHD.fetch.LHDTimeseriesDataFetcher',
  'lhdtmpdata': '/tmp/LHDtmpdata',
  'local_diag_path': 'None',
  'my_tmp': '/tmp'}

The properties from HJfetcher down come from the [DEFAULT] section, and
most of them are defined for use in substitutions (below).

.. _substitutions:

Simplifying changes by substitution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The syntax %(sym)s will substitute the contents of sym.  e.g.::

 fetchr =  pyfusion.acquisition.H1.fetch.H1LocalTimeseriesDataFetcherh1datafetcher
 data_fetcher = %(fetchr)s

Overriding substitutions
~~~~~~~~~~~~~~~~~~~~~~~~
cfg files read subsequently will override substitutions.  
Values to be substituted should be defined (in a safe way) in files
that use those substitutions.  Files loaded subsequently can then
override.
e.g. - in the main config file, put mytmp = /tmp
then   mytmp = $HOME/temp             
will override


Syntax
~~~~~~

This way only one edit needs to be made to change all diagnostics, if
the definition is fetchr is in the special [DEFAULT] section.

(From the 2.7 docs: 3 is a little different and cleaner)

The configuration file consists of sections, led by a [section] header
and followed by name: value entries, with continuations in the style
of RFC 822 (see section 3.1.1, “LONG HEADER FIELDS”); name=value is
also accepted. Note that leading whitespace is removed from
values. The optional values can contain format strings which refer to
other values in the same section, or values in a special DEFAULT
section. Additional defaults can be provided on initialization and
retrieval. Lines beginning with '#' or ';' are ignored and may be used
to provide comments.  Inline comments are should be avoided, and are
not accepted in the pyfusion python 3 version.

 .. _testing-config:

Testing config file behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Importing pyfusion automatically reads several files, so the way to
test is to start by clearing, *then* reading::
>>> cd pyfusion/test
>>> pyfusion.conf.utils.clear_config()
>>> pyfusion.read_config(["test1.cfg"])
# files ending in e should produce errors 
# this one has a substitution referencing an option defined in global
>>> pyfusion.read_config(["test1e.cfg"])

>>> pyfusion.conf.utils.clear_config()
>>> pyfusion.read_config(["test1.cfg"])
# the substitution in test2a (bar2a) overrides the initial one
>>> pyfusion.read_config(['test2a.cfg'])

>>> pyfusion.conf.utils.get_config_as_dict('Diagnostic','H1_multi')
 {'channel_1': 'H1MP',
  'channel_2': '-H1MP',
  'data_fetcher': 'pyfusion.acquisition.base.MultiChannelFetcher',
  'foo': 'bar2a',
  'other_attr': 'other',
  'some_attr': 'bar2a'}


User Defined Sections
~~~~~~~~~~~~~~~~~~~~~
Under test is a section [Plots] containing things like

``FT_Axis = [0, 0.08, 0, 300000]``

to provide defaults for the Frequency-Time axis etc

Note that such settings are highly dependent on the fusion experiment
and although they will be recognised in the code, they usually should
not be given values in code distributions.

The user could put their own items in there or other sections to avoid 

