""" This is the first working version with little selectivity over what analysis is done.
plots=0,  debug=0 has the least overhead but still slow even for 0.1 seconds
_PYFUSION_TEST_@@time_range=[4.6,4.6001]


"""
import pyfusion
from matplotlib import pyplot as plt
import numpy as np
from leastsq_Langmuir import leastsq, residuals, LPchar
from collections import OrderedDict as Ordict

_var_defaults=""" """
shot_number = [20181018, 6]
ROI = '4.6 4.601 1e-7'
dev_name = "W7M"
time_range = None
debug = 1
numplots = 3    # number of IV fits plotted
numshades = 20  # number of shaded clock cycles
mode = 'Te'     # Te, Isat, Vf
plots = 0  # controls plot complexity
maxlen = 20000  # limit detailed plots unless plots > 0
order = dict(Isat=0, Te=1, Vf=2)

exec(_var_defaults)
from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if time_range is not None:
    ROI = ' '.join([str('{t:.6f}'.format(t=t)) for t in time_range])
pyfusion.config.set('Acquisition:W7M', 'ROI', ROI)

dev = pyfusion.getDevice("W7M")
ip_data = dev.acq.getdata(shot_number, 'W7M_MLP_I')
vp_data = dev.acq.getdata(shot_number, 'W7M_MLP_U')
isat_data = dev.acq.getdata(shot_number, 'W7M_MLP_IS')
ck_data = dev.acq.getdata(shot_number, 'W7M_MLP_CK')
err_data = dev.acq.getdata(shot_number, 'W7M_MLP_ERR')
# if time range has a 3rd element, it is an ROI, and the bounds have been set
#  assuming we are not using an npz.
if time_range is not None and len(time_range) == 2:
    for dat in [ip_data, vp_data, isat_data, ck_data, err_data]:
        dat.reduce_time(time_range, copy=False)

if (len(ip_data.timebase) > maxlen) and plots > 1:
    wait_for_confirmation('Detailed plots of {n} points will be slow!'
                          .format(n = len(ip_data.timbase)))
plt.figure()
if plots > 0:
    ck_data.plot_signals(marker='.', color='orange', label='clock')
    isat_data.plot_signals(marker='.', label='all isat')
axt = plt.gca()
ips = ip_data.signal
vps = vp_data.signal
err_sig = err_data.signal

tb = ip_data.timebase
dt = np.diff(tb).mean()
thr = ck_data.signal.mean()
w = np.where(ck_data.signal > thr) # falling_edges are the transitions
falling_edges = w[0][np.where(np.diff(w[0]) > 2)[0]]
w = np.where(ck_data.signal < thr) # rising for the safe measurement period
rising_edges = w[0][np.where(np.diff(w[0]) > 2)[0]]
if falling_edges[0] < rising_edges[0]:  # make sure first is a rise
    falling_edges = falling_edges[1:]
if len(rising_edges) > len(falling_edges):
    rising_edges = rising_edges[0:-1]
ns = falling_edges[1] - falling_edges[0]
print('\nperiod of 2x clock = {ns} samples, complete cycle in {us:.1f}us'
      .format(ns=ns, us=ns*3*(tb[1]-tb[0])*1e6))
starts = rising_edges
ends = falling_edges

ip_means = [np.mean(ips[starts[i]:ends[i]]) for i, st in enumerate(starts)]
vp_means = [np.mean(vps[starts[i]:ends[i]]) for i, st in enumerate(starts)]
tb_means = [np.mean(tb[starts[i]:ends[i]]) for i, st in enumerate(starts)]
# err signal needs care as there is contamination due to the round trip delay
# and to a lesser extent, the prompt signal.
delay = int(600e-9/dt)
err_means = [np.mean(err_sig[starts[i]+delay:ends[i]]) for i, st in enumerate(starts)]

axt.plot(tb_means, -1 * np.array(ip_means), 'or', label='mean(-ip): c.f. isat', markersize=8)
axt.plot(tb_means, np.array(vp_means)/200, '.g', label='mean(vp(t))/200')

if debug > 0:
    if (plots < 1) and (len(tb) > maxlen):
        raise ValueError('need plots >= 1 or fewer than {ml:,} points for detailed errors'
                         .format(ml = maxlen))
    axt.plot(tb[w], -ips[w], '.m', label='-ip(t) during clock', markersize=3)
    for i, st in enumerate(starts):
        axt.plot(tb[starts[i]:ends[i]], -ips[starts[i]:ends[i]], '.r', markersize=6,
                 label=['-ip(t) used in mean', '_'][i > 0])
num_sets = len(ip_means) // 3
ip_grouped = np.reshape(ip_means[0:num_sets * 3], [num_sets, 3])
sat_set = np.argmax(-np.sum(ip_grouped, axis=0))
this_set = sat_set + order[mode]
# choose start so i_ion was first (i.e. grouped[:,0]) - phase order is isat, Te, vf
n_used = num_sets - 3
ip_grouped = np.reshape(ip_means[this_set:this_set + n_used * 3], [n_used, 3])
vp_grouped = np.reshape(vp_means[this_set:this_set + n_used * 3], [n_used, 3])
tb_grouped = np.reshape(tb_means[this_set:this_set + n_used * 3], [n_used, 3])
err_grouped = np.reshape(err_means[this_set:this_set + n_used*3], [n_used, 3])
for name, g in order.items():
    gcorr = (g + 3 - order[mode]) % 3  # correct for offset due to choice of primary err
    axt.step(tb_grouped[:, gcorr], err_grouped[:, gcorr], where='post',label='avg {nm} err'.format(nm=name))
    col = axt.get_lines()[-1].get_color()
    for cyc in range(3):
        i1 = cyc * 3  + gcorr + this_set
        axt.plot(tb[starts[i1]+delay:ends[i1]], err_sig[starts[i1]+delay:ends[i1]],lw=2, color=col)

fits = []
IV_plots = 0
vrange = [vp_grouped.min(), vp_grouped.max()]
vspan = np.abs(np.diff(vrange))
vrange = [vrange[0] - vspan/5, vrange[1] + vspan/10]

for tbg, ipg, vpg in zip(tb_grouped, ip_grouped, vp_grouped):
    guess = (max(vpg) - min(vpg))/3, vpg[order['Vf'] - order[mode]- sat_set + 1], np.mean(np.abs(ipg))
    (Te, Vf, i0), flag = leastsq(residuals, guess, args=(ipg, vpg), maxfev=40)
    if debug > 2:
        print('{f}: dV = {dV:.2f}, Te={Te:.2f}, Vf={Vf:.2f}, i0={i0:.3f},'  # offs={offs:.2f}, i_imbalance={im:.1f}%'
              .format(f=flag, dV=vp[0] - vp[-1], Te=Te, Vf=Vf, i0=i0))  # offs=offs, im = 100*i_p.mean()/np.max(np.abs(i_p))))
    fits.append([tbg[0], Te, Vf, i0])
    if (debug > 0) and (IV_plots < numplots):
        if IV_plots == 0:
            fig,axc = plt.subplots(1,1)
        IV_plots += 1
        vv = np.linspace(vrange[0], vrange[1], 1000)
        axc.plot(vpg, ipg, 'o')  #, label='After {n} iters'.format(n=nits))
        axc.plot(axc.get_xlim(), [0, 0], 'lightgray')
        print(Te, Vf, i0)
        print(ipg)
        axc.plot(vv, LPchar(vv, Te, Vf, i0),
                 label='fitted LP char {tm:.4f}s, Te={te:.0f}, Vf={vf:.1f}'
                 .format(tm=fits[-1][0], te=fits[-1][1], vf=fits[-1][2]))
        axc.legend(loc='best',prop={'size': 'small'})


fits = np.array(fits, dtype=np.float32)
wnan = np.where((fits[:,-1] > 0) | (fits[:,-1] < -1))[0]
fits[wnan,-1] = np.nan
axt.step(fits[:, 0], -fits[:, -1], where='mid',label='isat_fitted')
if order[mode] > 0: # plot the fitted quantity if not isat
    ind = -2 if order[mode] == 2 else -3
    axt.step(fits[:, 0], fits[:, ind], where='mid',label='{m}_fitted'.format(m=mode))
axt.set_ylim(0, axt.get_ylim()[1])
from matplotlib.patches import Rectangle
ylims = axt.get_ylim()
for i in range(min(numshades, len(starts))):
    lleft = (tb[starts[i]], ylims[0])
    fillcol = "lightcyan"
    axt.add_patch(Rectangle(lleft, tb[ends[i]]-tb[starts[i]] , ylims[1]-ylims[0],
                            facecolor=fillcol, edgecolor='lightgray',label=["'steady' area","_"][i>0]))

axt.legend(loc='best',prop={'size': 'small'})
plt.show(block=0)
