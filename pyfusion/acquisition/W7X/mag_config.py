""" Return configuration information as ratios or currents
"""
import pyfusion
from pyfusion.utils.time_utils import utc_ns
import os
import matplotlib.pyplot as plt
import numpy as np


from sqlalchemy import create_engine 
# put the file in the example (this) folder although it may be better in the acquisition/W7X folder
dbpath = os.path.dirname(__file__)
engine=create_engine('sqlite:///'+ dbpath + '/W7X_mag_OP1_1.sqlite', echo=False)
conn = engine.connect()

def plot_trim(ax, mag_dict=None, offset=[1.08, 0.88], size=.06, aspect=1, **pltkwargs):
    
    if mag_dict is None:
        return
    md = mag_dict
    tc = np.array([md['ITrim_{i}'.format(i=i)] for i in range(1,6)])
    tc = size * (1 + 5*tc/md['INonPlanar_1'])
    Z = tc * np.exp(2*np.pi*1j*np.arange(1,6)/5)
    rot = np.exp(-np.pi*3./10j)  # -np.pi/10j for 1 at top, -np.pi*3./10 for mid 1-5 at top
    x, y = [offset[0] + np.real(Z*rot), offset[1] + aspect*np.imag(Z*rot)]
    xl, yl = x.tolist(), y.tolist()
    for ii, [xli, yli] in enumerate(zip(xl, yl)):
        ax.plot([offset[0], xli], [offset[1], yli], transform=ax.transAxes, clip_on=0, 
                **pltkwargs)
        if ii+1 in [1,5]:
            ax.text(xli, yli, str(ii+1), transform=ax.transAxes, clip_on=0, 
                    bbox=[{}, dict(facecolor='green', alpha=0.2)][ii+1 == 5])
    for xxx in xl,yl:
        xxx.append(xxx[0])
    ax.plot(xl, yl, transform=ax.transAxes, clip_on=0, lw=0.5, **pltkwargs)
    # ax.plot([1.1,1.2],[0.9,0.2],transform=axu.transAxes,clip_on=0)

def get_mag_config(shot, from_db=True, ratios=True):
    dev = pyfusion.getDevice('W7X')
    currents = []
    try:
        if from_db:
            coil_list = ',' + ','.join('ITrim_{i}'.format(i=i) for i in range(1, 6))
            coil_list += ',' + ','.join('IPlanar_{a}'.format(a=a) for a in ['A', 'B'])
            coil_list += ',' + ','.join('INonPlanar_{i}'.format(i=i) for i in range(1, 6))
            coil = coil_list  # convenience for error message
            result = conn.execute('select shot' + coil_list + ' from summ where shot={shot}'
                                  .format(shot=shot[0]*1000 + shot[1]))
            data = result.fetchone()
            ddat = dict(data)
            if  ddat['INonPlanar_1'] is None:
                pyfusion.logger.warning('Guessing NonPlanar currents')
                ddat['INonPlanar_1'] = 12400. 
                ddat['IPlanar_B'] = 4836.

            ddat.update(dict(ratio = ddat['IPlanar_B']/ddat['INonPlanar_1']))
            ref = ddat['INonPlanar_1']
            return([data.INonPlanar_1, ddat['IPlanar_B']/ref, ddat['IPlanar_B']/ref],  ddat)

        else:
            for i, coil in enumerate(['NonPlanar_1', 'Planar_A', 'Planar_B']):
                data = dev.acq.getdata(shot, 'W7X_I'+coil)
                currents.append(np.average(data.signal))
                if ratios and coil is not 'NonPlanar_1':
                    currents[i] = currents[i]/currents[0] # no point rounding - still get 0.390000001
            return currents, None #  None instead of the dictionary
    except Exception as reason:
        print('data not found for coil {coil}, shot {shot} \n{r}'
              .format(coil=coil, shot=shot, r=reason))
        return None, None

print(get_mag_config([20160310, 7]))
