Logging
=======

pyfusion uses the standard python loggers.
The default is to log info and above (not debug) into a file, and
warning and above to the console.

The configuation is changed by the logging.cfg file, whose contents
are described in 

https://docs.python.org/2/library/logging.config.html#logging-config-fileformat

partial example:
# this connect both those described below
[logger_root]
level=DEBUG
handlers=consoleHandler, fileHandler

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('pyfusion.log', 'w')  

[handler_consoleHandler]
class=StreamHandler
level=WARNING
formatter=simpleFormatter
args=(sys.stdout,)

