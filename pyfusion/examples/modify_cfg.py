import numpy as np
debug=0  # 1 will stop at error

dat = np.loadtxt('LP_coords.txt', dtype=[('seg', 'i'), ('name', 'S24'), ('X', 'f'), ('Y', 'f'), ('Z', 'f')],
                 delimiter='\t')
with open('pyfusion/pyfusion.cfg') as cf:
    clines=cf.readlines()

print('clines initially {l} lines'.format(l=len(clines)))

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
        l = np.where((dat['seg'] == seg) & (dat['name'] == 'Sonde_{p}.1 (Spitze)'.format(p=probe)))[0]
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
        line = str('W7-X-Koord = {x:.5f}, {y:.5f}, {z:.5f}\n'
                   .format(x=dat['X'][l]/1e3, y=dat['Y'][l]/1e3, z=dat['Z'][l]/1e3))
        clines.insert(i+1, line)
        i += 1

print('final length {l}'.format(l=len(clines)))

with open('pyfusion.cfg.new','wb') as outfile:
    outfile.writelines(clines)

# W7-X Koordinatensystem
