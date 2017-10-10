import numpy as np
debug=0  # 1 will stop at error

""" Orig
dat = np.loadtxt('LP_coords.txt', dtype=[('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')],
                 delimiter='\t')
"""
# uncomment one of these
attr='Positions of probe centre'

if 'area' in attr:
    attr_filename = '/data/databases/W7X/LP/merge_in_probe_area.txt'
    attr_dtype = [('name', 'S24'), ('area', 'f')]
    attr_fmt = 'area = {a:.3f}e-06\n'  # fake it so result always has power -6
    dtype=[('name','S20'), ('area','f')]
    def attr_dict(dat):
        return(dict(a=dat['A'])) # mm2 here, m2 in file (see -6 above)
else:
    attr_filename = 'LP_coords.txt'
    attr_dtype = [('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')]
    attr_fmt = 'coords_w7_x_koord = {x:.5f}, {y:.5f}, {z:.5f}\n'
    dtype=[('name','S20'), ('X','f'), ('Y','f'), ('Z','f')]
    def attr_dict(dat):
        return(dict(x=dat['X']/1e3, y=dat['Y']/1e3, z=dat['Z']/1e3))

attr_filename = '/data/databases/W7X/OP1.2a/TDU_info/TDU_Langmuir_properties_KDM001PLC-750_BPC_KJM001_25.txt'

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


if 'TDU' in attr_filename:
    TDU_dat = TDU_info()
    dat = TDU_dat.getattr(attr, dtype=dtype)
else:
    dat = np.loadtxt(attr_filename, dtype=attr_dtype, delimiter='\t')

with open('pyfusion/pyfusion.cfg') as cf:
    clines=cf.readlines()

initial = len(clines)
print('clines initially {l} lines'.format(l=initial))

i = 0
while i < len(clines)-1:
    while (i < len(clines)-1) and ((clines[i][0] == '#') or \
                                   (('Diagnostic:W7X' not in clines[i]) and \
                                   ('DTU' not in clines[i] or 'W7X_L5' not in clines[i]))):
        i += 1

    if (clines[i].strip()[-3:] != '_I]') or ('TDU' not in clines[i]):
        print('skipping line {i} (no _I): {l}'.format(i=i, l=clines[i]))
        i += 1
    else:
        toks = clines[i].split('_')
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
            #  51 is the Upper TDU
            Qname = 'QRP5{s}CE1{probe:02d}'.format(s=[0,1][seg=='U'], probe=probe)
            l = np.where(dat['name'] == Qname)[0]
        else:
            l = np.where(dat['name'] == clines[i].split(':')[1].split(']')[0])[0]

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

        clines.insert(i+1, line)
        i += 1

print('final length {l}, {e} extra'.format(l=len(clines),e=len(clines)-initial))

with open('pyfusion.cfg.new','wb') as outfile:
    outfile.writelines(clines)

# W7-X Koordinatensystem

