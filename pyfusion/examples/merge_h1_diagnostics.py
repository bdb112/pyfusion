import MDSplus as MDS
#import pyfusion
import numpy as np
import pylab as pl
from matplotlib.mlab import stineman_interp

_var_default="""

debug=0
verbose=1
exception = Exception
# this doesn't help - need a lookup table.
diags='b_0,k_h,ne_1,ne_4,ne_7,p_rf,phd_rf,flow_1,flow_2,flow_3'
"""

# assume dd contains data dictionary

exec(_var_default)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

t_mid = dd['t_mid']
shot = dd['shot']
for k in diags.split(','):
    if k not in dd.keys():
        dd[k] = t_mid.astype(float)*0

good,bad = [],[]

for sh in np.unique(shot):
    try:
        h1tree = MDS.Tree('h1data',sh)
        try:
            ne_node = h1tree.getNode('.electr_dens.ne_het:ne_1')
            num_ne = 3
        except:
            ne_node = h1tree.getNode('.electr_dens.ne_het:ne_centre')
            num_ne = 1

        ne_1 = ne_node.data()
        ne_time = ne_node.dim_of().data()
        if num_ne>1:
            ne_4 = h1tree.getNode('.electr_dens.ne_het:ne_4').data()
            ne_7 = h1tree.getNode('.electr_dens.ne_het:ne_7').data()

        main_current=h1tree.getNode('.operations.magnetsupply.lcu.setup_main:I2').data()
        b_0 = float(main_current)/13888.
        is2 = h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I2').data()
        is3 = h1tree.getNode('.operations.magnetsupply.lcu.setup_sec:I3').data()
        sweeping = is2 != is3
        if sweeping:
            ihel_node = h1tree.getNode('.operations:i_hel')
            k_h_time = ihel_node.dim_of().data()
            k_h = 1000*((2.17*ihel_node.data()-0.070)
                        /float(main_current))
        else:
            k_h = float(is2)/float(main_current)

        try:
            ph2 = h1tree.getNode('.LOG.HEATING.SNMP.T2.OPERATIONAL.LLRF.STALLRFPHD').data()
            ph1 = h1tree.getNode('.LOG.HEATING.SNMP.T1.OPERATIONAL.LLRF.STALLRFPHD').data()
            phd_rf = ph2-ph1
        except:
            phd_rf = phd_rf+np.nan
        rf_drive_node = h1tree.getNode('.RF.rf_drive')
        p_rf = 8*rf_drive_node.data()**2
        p_rf_time = rf_drive_node.dim_of().data()
        flow_1 = h1tree.getNode('.operations.magnetsupply.lcu.log:gas1_set').data()
        flow_2 = h1tree.getNode('.operations.magnetsupply.lcu.log:gas2_set').data()
        flow_3 = h1tree.getNode('.operations.magnetsupply.lcu.log:gas3_set').data()
        good.append(sh)

    except exception as reason:
        bad.append(sh)
        print('failed to open shot {sh}: {r} {a}'.format(sh=sh, r=reason, a=reason.args))
        continue

    w = np.where(sh == shot)[0]
    times = t_mid[w]
    for ne in 'ne_1,ne_4,ne_7'.split(',')[0:num_ne]:
        valarr = stineman_interp(times, ne_time, eval(ne))
        dd[ne][w] = valarr

    if sweeping:
        k_h = stineman_interp(times, k_h_time, k_h)
    dd['k_h'][w] = k_h

    dd['p_rf'][w] = stineman_interp(times, p_rf_time, p_rf)
    for k in 'b_0,phd_rf,flow_1,flow_2,flow_3'.split(','):
        dd[k][w] = eval(k)
    

print('{g} shots successful, {b} shots failed:'.format( g=len(good), b=len(bad)))
print(bad)
          
