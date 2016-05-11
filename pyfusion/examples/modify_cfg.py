import numpy as np
debug=0  # 1 will stop at error

""" Orig
dat = np.loadtxt('LP_coords.txt', dtype=[('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')],
                 delimiter='\t')
"""
# uncomment one of these
"""
attr_filename = 'LP_coords.txt'
attr_dtype = [('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')]
attr_fmt = 'coords_w7-x-koord = {x:.5f}, {y:.5f}, {z:.5f}\n'
def attr_dict(dat):
    return(dict(x=dat['X']/1e3, y=dat['Y']/1e3, z=dat['Z']/1e3))

"""

attr_filename = '/data/databases/W7X/LP/merge_in_probe_area.txt'
attr_dtype = [('name', 'S24'), ('A', 'f')]
attr_fmt = 'area = {a:.3f}e-06\n'  # fake it so result always has power -6
def attr_dict(dat):
    return(dict(a=dat['A'])) # mm2 here, m2 in file (see -6 above)

dat = np.loadtxt(attr_filename, dtype=attr_dtype, delimiter='\t')
with open('pyfusion/pyfusion.cfg') as cf:
    clines=cf.readlines()

initial = len(clines)
print('clines initially {l} lines'.format(l=initial))

i = 0
while i < len(clines)-1:
    while (i < len(clines)-1) and ((clines[i][0] == '#') or ('Diagnostic:W7X_L' not in clines[i])):
        i += 1

    if clines[i].strip()[-3:] != '_I]':
        print('skipping line {i} (no _I): {l}'.format(i=i, l=clines[i]))
        i += 1
    else:
        toks = clines[i].split('_')
        seg = int(toks[1][-1])
        probe = int(toks[2][2:])
        print(i, seg, probe)
        if 'coord' in attr_filename:
            l = np.where((dat['seg'] == seg) & (dat['name'] == 'Sonde_{p}.1 (Spitze)'.format(p=probe)))[0]
        else:
            l = np.where(dat['name'] == clines[i].split(':')[1].split(']')[0])[0]

        if len(l) != 1:
            msg = str('finding segment {s}, probe {p}'.format(s=seg, p=probe))
            if debug: 
                raise LookupError(str)
            else:
                print('Warning' + msg)
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
