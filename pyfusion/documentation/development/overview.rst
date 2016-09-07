Pyfusion Development
====================

-----------------
Design principles
-----------------

Documentation
-------------

Documentation is maintained in the code repository. An online version is kept up to date `here <http://h1nf.anu.edu.au/collaborate/pyfusion/docs/>`_


Test-driven design (TDD)
------------------------


Distributed source code
-----------------------


------------------------
Developing with pyfusion
------------------------

Documentation
-------------

* use sphinx
* built docs not stored in repository

Decorators/Metaclasses
----------------------

Used to populate objects such as <data> with functions such as <fetch> that knows the details of the particular plasma device.  see "register"

MetaClasses

https://jakevdp.github.io/blog/2012/12/01/a-primer-on-python-metaclasses/

Intro to decorators

https://realpython.com/blog/python/primer-on-python-decorators/

Use as function factory

https://jeffknupp.com/blog/2013/11/29/improve-your-python-decorators-explained/


Tests
-----
* use nosetests

* running nosetest pyfusion should be *very* fast. The idea behind regular testing is that the tests should be so fast that you don't hesitate to run the test. Any test which requires significant computation or hard disk / network access should be disabled by default. Using $HOME/.pyfusion/tests.cfg you can enable any of these tests when you need them.

* selection of which tests are run is done with nosetest attributes, `see nose docs for detail <http://somethingaboutorange.com/mrl/projects/nose/0.11.2/plugins/attrib.html>`_. For example, to run all tests except those which use SQL::

   > nosetests -a '!sql' pyfusion

* There should not be conditional tests for pyfusion.USE_ORM within the test code as there can be confusion as to which configuration settings are present in the testing environment. Instead, use a separate class for the SQL code and provide it with the 'sql' attribute.

The available attributes are:

========  =========================================================================
``sql``   test requires sqlalchemy module
``net``   test requires internet access
``lhd``   test connects to LHD data acquisition system
``tjii``  test connects to TJII data acquisition system
``h1``    test connects to H-1 data acquisition system
``plot``  test requires matplotlib module 
``daq``   test connects to a data system (superset of ``lhd``, ``tjii`` and ``h1``)
========  =========================================================================


Python 3 Issues
===============

pickling and unicode
--------------------

Hard to find settings that allow a python 2 picklw to be read by python 3.
http://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3

Perhaps it better to use json if possible.


