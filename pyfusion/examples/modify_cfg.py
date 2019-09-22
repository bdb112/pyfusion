""" The previous version works for W7X Michael's probe file.  This one is for the BES coords at HeliotronJ
    Hopefully the previous functionality still works...
    ps: The area values were not added in the previous version....
"""

import numpy as np
import re

debug=0  # 1 will stop at error

""" Orig
dat = np.loadtxt('LP_coords.txt', dtype=[('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')],
                 delimiter='\t')
"""

# uncomment one set of these
attr_filename = '/data/databases/W7X/OP1.2a/TDU_info/TDU_Langmuir_properties_KDM001PLC-750_BPC_KJM001_25.txt'
target = '^\[Diagnostic:W7X_[U,L]TDU_LP[0-2][0-9]_I\]'

attr = 'projected area'  #these two or
attr_dtype=[('name','S20'), ('X','f'), ('Y','f'), ('Z','f')]

#attr = 'Positions of probe centre' #these two
#attr_dtype=[('name','S20'), ('area','f')]

"""
# Limiter
target = '^\[Diagnostic:W7X_L5[3,5]_LP[0-2][0-9]_I\]'
attr = 'Lim_area'
attr_filename = '/data/databases/W7X/LP/merge_in_probe_area.txt'

attr = 'Lim_tip_coords'
attr_filename = 'LP_coords.txt'
attr_dtype = [('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')]
attr_fmt = 'coords_w7_x_koord = {x:.5f}, {y:.5f}, {z:.5f}\n'
"""

attr = 'coordinates of volume centre'
target = '^\[Diagnostic:HeliotronJ_BES1_[0-2][0-9]\]'
attr_filename = 'pyfusion/acquisition/HeliotronJ/BES_data.txt'
attr_dtype=[('name','S20'), ('X','f'), ('Y','f'), ('Z','f')]
attr_fmt = 'coords_reduced = {x:.5f}, {y:.5f}, {z:.5f}\n'

# attr = 'quasi toroidal coords QXM41'  # just a line to indicate start of this data
attr = 'fake coords_reduced QXM41'  # just a line to indicate start of this data
target = '^\[Diagnostic:W7X_MIR_[4][0-9][0-9][0-9]\]'
attr_filename = 'pyfusion/acquisition/W7X/QXM41_data.txt'
attr_dtype=[('name','S20'), ('X','f'), ('Y','f'), ('Z','f')]
attr_fmt = 'coords_reduced = {x:.5f}, {y:.5f}, {z:.5f}\n'

# Note: pyfusion.cfg file has the x,y,z coords, but is used at present with
# equal intervals in theta as a first approx via /QXM41_data.txt'  
# attr = 'quasi toroidal coords QXM41'  # just a line to indicate start of this data
attr = 'coords_w7-x-koord for QXM41 from Kian 9/2019'  # just a line to indicate start of this data
target = '^\[Diagnostic:W7X_MIR_[4][0-9][0-9][0-9]\]'
attr_filename = 'pyfusion/acquisition/W7X/QXM_xyz_data.txt'
attr_dtype=[('name','S20'), ('X','f'), ('Y','f'), ('Z','f')]
attr_fmt = 'coords_w7-x-koord = {x:.5f}, {y:.5f}, {z:.5f}\n'


if 'coords_reduced' in attr:
    def attr_dict(dat):
        r2d = 180/np.pi
        return(dict(x=dat['X']/r2d, y=dat['Y']/r2d, z=dat['Z']/r2d))
elif 'oord' in attr:
    def attr_dict(dat):
        return(dict(x=dat['X']/1e3, y=dat['Y']/1e3, z=dat['Z']/1e3))
elif 'area' in attr:
    attr_dtype = [('name', 'S24'), ('area', 'f')]
    attr_fmt = 'area = {a:.3f}e-06\n'  # fake it so result always has power -6
    def attr_dict(dat):
        return(dict(a=dat['area'])) # mm2 here, m2 in file (see -6 above)

class TDU_info():
    def __init__(self, filename=attr_filename):
        self.filename = filename
        with open(filename) as opened_file:
            self.buffer = opened_file.readlines()

    def getattr(self, attr='projected area', dtype=[('name','S20'), ('area','f')]):
        headings = [l for l, buf in enumerate(self.buffer) if attr in buf]
        if len(headings) is not 1:
            raise LookupError("can't find or too many {attr} in {f}"
                              .format(attr=attr, f=self.filename))
        blanks = [b for b, buf in enumerate(self.buffer[headings[0]:]) if buf.strip() == '']
        if len(blanks) < 1:
            raise LookupError("can't find blanks in {f} after {attr}"
                              .format(attr=attr, f=self.filename))
        return(np.genfromtxt(self.buffer[headings[0]+1: headings[0]+blanks[0]], dtype=dtype)) #, dtype=


if 'TDU' or 'BES' in attr_filename:
    TDU_dat = TDU_info(attr_filename)
    dat = TDU_dat.getattr(attr, dtype=attr_dtype)
else:
    dat = np.loadtxt(attr_filename, dtype=attr_dtype, delimiter='\t')

with open('pyfusion/pyfusion.cfg') as cf:
    clines=cf.readlines()

initial = len(clines)
print('clines initially {l} lines'.format(l=initial))

i = 0
while i < len(clines)-1:
    #while (i < len(clines)-1) and ((clines[i][0] == '#') or \
    #                               (('Diagnostic:W7X' not in clines[i]) and \
    #                               ('DTU' not in clines[i] or 'W7X_L5' not in clines[i]))):
    while not re.match(target, clines[i]):
       i += 1
       if (i >= len(clines)):
           break
    else:
        if 'BES' not in attr_filename and 'QXM' not in attr_filename:
            toks = clines[i].strip().split('_')
            if 'TDU' in attr_filename:
                seg = toks[1][0]
            else:
                seg = toks[1][-1]

            probe = int(toks[2][2:])
        
            print(i, seg, probe)
            if 'coord' in attr_filename:
                l = np.where((dat['seg'] == seg) & (dat['name'] == 'Sonde_{p}.1 (Spitze)'.format(p=probe)))[0]
            elif 'TDU' in attr_filename:
                print(seg, i)
                #  51 is the Upper TDU - not sure if I used the QRP5 ever?
                Qname = 'QRP5{s}CE1{probe:02d}'.format(s=[0,1][seg=='U'], probe=probe)
                l = np.where(dat['name'] == Qname)[0]
            else:
                l = np.where(dat['name'] == clines[i].split(':')[1].split(']')[0])[0]
        elif 'QXM' in attr_filename:
            seg = None
            probe = clines[i].strip().split('tic:', 1)[1].split(']')[0]
            Qname = 'QXM{hs}CE{pnum}0'.format(hs=probe[-4:-2], pnum=probe[-2:])
            l = np.where(dat['name'] == Qname)[0]
        else:
            probe = clines[i].strip().split('tic:', 1)[1].split(']')[0]
            if 'BES' in attr_filename:
                probe = probe.split('_')[1]
            seg = None
            l = np.where(dat['name'] == probe)[0]
            
        # print(len(dat))
        if len(l) != 1:
            msg = str('problem finding segment {s}, probe {p}'.format(s=seg, p=probe))
            if debug: 
                raise LookupError(msg)
            else:
                print('Warning ' + msg)
                i += 1
                continue
        else:
            l = l[0]
        """ simple, old way
        line = str('coords_w7-x-koord = {x:.5f}, {y:.5f}, {z:.5f}\n'
                   .format(x=dat['X'][l]/1e3, y=dat['Y'][l]/1e3, z=dat['Z'][l]/1e3))
        """
        line = str(attr_fmt.format(**attr_dict(dat[l])))
        print(l, line)

        clines.insert(i+1, line)
        i += 1

print('final length {l}, {e} extra'.format(l=len(clines),e=len(clines)-initial))

with open('pyfusion.cfg.new','wb') as outfile:
    outfile.writelines(clines)

print(outfile.name)
# W7-X Koordinatensystem

