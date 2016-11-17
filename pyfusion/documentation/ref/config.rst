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

Valid Dates
-----------
A new feature allows configuration to change for different date
ranges.  Initially the dates work back from the latest config.  If for
a particular diagnostic, the date is outside the valid_dates, then
alternate diag names such as W7XM1_L53_LP02_I are checked for in the
config file.  If found, and the date range matches, we are finished. 
Otherwise an error is generated.

A second modification (M2) builds on the first (M1), so the effect is
cumulative.  If a diagnostic is missing on a day, it will have to be
restored on the previous day. Diagnostics can be suppressed for now by
setting DMD=0, so all the other charactersitics remain, so it can be
easily restored.



Loading config files
--------------------
When pyfusion is imported, it will load the default configuration file
provided in the source code (that is in the pyfusion directory)
followed by your custom configuration file, 
in ``$HOME/.pyfusion/pyfusion.cfg``, if it exists. 
and finally files pointed to by the environment variable PYFUSION_CONFIG_FILE
if they exist. This allows temporarily overriding config variables.

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

See :ref:`testing-config`

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

A seperate configuration file "tests.cfg", in the same ".pyfusion" folder in your home directory, can be used during development to enable tests which are disabled by default.

An example of the syntax is::

	[EnabledTests]
	mdsplus = True
	flucstrucs = True

etc...




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

