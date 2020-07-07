import sys
#_PYFUSION_TEST_@@W7XNET
sys.path.insert(0, 'c:\\PortablePrograms\\winpythonlibs')
import MDSplus as MDS
from matplotlib import pyplot as plt

srv = MDS.Connection('mds-data-1')
tr = srv.openTree('QRN', 160309032)
plt.plot(srv.get('.HARDWARE:ACQ132_170:INPUT_15',label='20160309_32'))
tr = srv.openTree('QRN', 160309013)
plt.plot(srv.get('.HARDWARE:ACQ132_170:INPUT_15',label='20160309_13'))
plt.legend(loc='best')
plt.show()
