import os, logging
import logging.config

from pyfusion.conf import PyfusionConfigParser
from pyfusion.conf.utils import read_config
from pyfusion.orm import ORMManager
from pyfusion.version import get_version

# This grabs the directory of the pyfusion module.
PYFUSION_ROOT_DIR = os.path.dirname(__file__)

# Location of the logging configuration file.
LOGGING_CONFIG_FILE = os.path.join(PYFUSION_ROOT_DIR, 'conf', 'logging.cfg')

# Grab the pyfusion version
VERSION = get_version()

# This creates a parser to process and store configuration file data.
# PyfusionConfigParser is a subclass of ConfigParser.ConfigParser in the
# python standard library. It is customised to parse [Type:Name] sections
# in configuration files.
config = PyfusionConfigParser()

# This allows us to activate and de-activate the object relational
# mapping (ORM) code from within a python session, and also helps to
# keep the ORM code separate so it doesn't slow things down for
# users who don't want the database backend. 
orm_manager = ORMManager()

# This sets up an instance of logger from the python standard library.
logging.config.fileConfig(LOGGING_CONFIG_FILE)
logger = logging.getLogger("pyfusion")

# Find the default pyfusion configuration file...
DEFAULT_CONFIG_FILE = os.path.join(PYFUSION_ROOT_DIR, 'pyfusion.cfg')

# ... and the user's custom configuration file. First, if they don't
# already have a folder for pyfusion stuff then let's make one
USER_PYFUSION_DIR = os.path.join(os.path.expanduser('~'), '.pyfusion')
if not os.path.exists(USER_PYFUSION_DIR):
    os.mkdir(USER_PYFUSION_DIR)
# and here is the custom user configuration file
USER_CONFIG_FILE = os.path.join(USER_PYFUSION_DIR, 'pyfusion.cfg')

# Also allow specification of other configuration files from
# a PYFUSION_CONFIG_FILE environment variable
USER_ENV_CONFIG_FILE = os.getenv('PYFUSION_CONFIG_FILE','')
if ((len(USER_ENV_CONFIG_FILE) > 0) and
    not(os.path.exists(USER_ENV_CONFIG_FILE))): 
    raise IOError('Error - cfg file {f} pointed to by USER_ENV_CONFIG_FILE'
                  ' not found!  Check/delete the env var {v}'
                  .format(f=USER_ENV_CONFIG_FILE,v="PYFUSION_CONFIG_FILE"))


# Now we actually load the configuration files. Settings in
# DEFAULT_CONFIG_FILE will be superseded by those in USER_CONFIG_FILE,
# and USER_ENV_CONFIG_FILE will supersede both. Note that, so that
# DEFAULT_CONFIG can substitute local values, USER_CONFIG
# is read in initially as well.  In addition to reading in settings,
# read_config() will set up the ORM backend if required.
# Note that the $HOME/.pyfusion file overrides the $PYFUSION_ROOT_DIR file.
DEBUG = os.getenv('PYFUSION_DEBUG','0')  # probably better as an env var
try:   # if it looks like an int, make it one.
    DEBUG=int(DEBUG)
except:
    pass


def DBG():
    """ A function that returns pyfusion.DEBUG if it is and initeger else 0
    the hasattr function is faster than is_numlike
    143 ns per loop if int
    306 ns if string
    Can still use   if pyfusion.DBG() >  if speed is critical - problem is only 
    that any string in pyfusion.DEBUG will stop at that point too.
    """ 
    if hasattr(DEBUG, 'real'):
        return(DEBUG)
    else:
        return(0)

# VERBOSE is likely to be used as an env var for debugging and as a config
# var, depending on taste.
VERBOSE = int(os.getenv('PYFUSION_VERBOSE','0'))  # allows config info to be debugged
#read_config([DEFAULT_CONFIG_FILE, USER_CONFIG_FILE, USER_ENV_CONFIG_FILE])
""" 
Common stuff in DEFAULT_CONFIG_FILE, including my_tmp
Computer specific personalisation in USER_CONFIG_FILE
e.g. my_tmp = /tmp/blackwell
Conputer independent personalisation in  USER_ENV_CONFIG_FILE
the USER_CONFIG file (.pyfusion) must be after DEFAULT_CONFIG to override defaults
"""
read_config([DEFAULT_CONFIG_FILE, USER_CONFIG_FILE, USER_ENV_CONFIG_FILE])

def reload_config():
    """ convenience function to read in new config.  Not sure if it is kosher..
    So if in doubt, restart pyfusion.
    Fails to forget lines that were commented out
    maybe need also   config = PyfusionConfigParser() and return it
    but this is worse - commented out.
    """
    #config = PyfusionConfigParser()
    read_config([DEFAULT_CONFIG_FILE, USER_CONFIG_FILE, USER_ENV_CONFIG_FILE])
    #return(config)

# verbosity level from environment has priority, otherwise use config, which 
# defaults to 0.  Note that we looked at the env var previously to allow debug of config
VERBOSE = int(os.getenv('PYFUSION_VERBOSE',
                    config.get('global','VERBOSE',vars={'VERBOSE':'0'})))
## Variable precision - allows data sets to be much bigger - up to 4x  bdb Dec-2012

# This allows gain factors to be ignored, to see raw digitiser output
RAW = int(os.getenv('PYFUSION_RAW',
                    config.get('global','RAW',vars={'RAW':'0'})))

# if 1, will wget -x first to cache a local copy, then use it
CACHE = int(os.getenv('PYFUSION_CACHE',
                    config.get('global','CACHE',vars={'CACHE':'0'})))

# so far implemented for W7X url fetch_mode only
# will produce warning and try a second time for 3x the timeout value
TIMEOUT = int(os.getenv('PYFUSION_TIMEOUT',
                    config.get('global','TIMEOUT',vars={'TIMEOUT':'30'})))

# restrict the number of samples to speed up access over slow nets (for debugging of course)
NSAMPLES = int(os.getenv('PYFUSION_NSAMPLES',
                    config.get('global','NSAMPLES',vars={'NSAMPLES':'0'})))

root_dir = os.path.split(os.path.abspath( __file__ ))[0]

# cache control is likely to be computer dependent
try:
    TMP_FREE_BYTES = float(config.get('global','tmp_free_bytes'))
except:
    TMP_FREE_BYTES = 1e9
    print('defaulting [global]temp_free_bytes to {t:,}'.format(t=TMP_FREE_BYTES))

try:
    from numpy import dtype as npdtype
    # Beware!  vars takes  precedence over others!
    prec_med=npdtype(config.get('global','precision_medium')) #,vars={'precision_medium':'float32'}))
except:
    print('defaulting medium precision value, either because of error or'\
              ' missing value precision_medium in globals')
    from numpy import dtype as npdtype
    prec_med=npdtype('float32')

# warning - may need to move this earlier - for some reason the fftw3 import 
# doesn't work if it is ahead of this one.
try:
    COLORS=config.get('global','colors')
except:
    if VERBOSE>0: print('colors not in config file - no color!')
    COLORS=None

# 
default_fft_type = 'default numpy'
try:
    fft_type = config.get('global','fft_type')
except:
    fft_type = default_fft_type
    if VERBOSE>0: print('fft_type not in config - use default: {s}'.format(s=fft_type))

if fft_type == 'fftw3':  # Note - don't use quotes in .cfg file
    try:
        import pyfftw
    except:
        print('unable to load fftw3')
        fft_type = default_fft_type

print('Using {f} for FFT:  set with e.g. [global] fft_type=fftw3'.format(f=fft_type))

if fft_type == 'fftw3':
    try: 
        from pyfusion.utils.fftw3_bdb_utils import save_wisdom, load_wisdom
        load_wisdom()
        # note - see pyfusion.utils.primefactors for paste to get wisdom
    except:
        print('using fftw3, but no saved wisdom')
        
    #pyfftw.fftw_set_timelimit(10)  # limit time spent planning to 10 secs -
    #                                hopefully next time it will get further.

    try:
        fftw3_args=config.get('global','fftw3_limit')
        print('fftw3_args not yet implemented in config')
    except:
        if VERBOSE>0: print('fftw3_args not in config file e.g. planning time limit')
        fftw3_args = None

fftw3_args = dict(planning_timelimit=10.)
# pyfusion.fftw3_args= {'planning_timelimit': 50.0, 'threads':1, 'flags':['FFTW_MEASURE']}

# We import these into the base pyfusion namespace for convenience.
from pyfusion.devices.base import getDevice
from pyfusion.acquisition.utils import getAcquisition

