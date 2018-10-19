import pyfusion
from matplotlib import pyplot as plt
import numpy as np
from leastsq_Langmuir import leastsq, residuals, LPchar

_var_defaults=""" """
shot = [20181018, 6]
ROI = '4.6 4.601 1e-7'
dev_name = "W7M"
time_range = None
debug = 1
numplots =3

exec(_var_defaults)
from  pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if time_range != []:
    ROI = ' '.join([str('{t:.6f}'.format(t=t)) for t in time_range])
pyfusion.config.set('Acquisition:W7M', 'ROI', ROI)

dev = pyfusion.getDevice("W7M")
ip_data = dev.acq.getdata(shot, 'W7M_MLP_I')
vp_data = dev.acq.getdata(shot, 'W7M_MLP_U')
isat_data = dev.acq.getdata(shot, 'W7M_MLP_IS')
ck_data = dev.acq.getdata(shot, 'W7M_MLP_CK')
if time_range is not None:
    for dat in [ip_data, vp_data, isat_data, ck_data]:
        dat.reduce_time(time_range, copy=False)

isat_data.plot_signals(marker='.', label='all isat')
ck_data.plot_signals(marker='.', color='orange', label='clock')
axt = plt.gca()
ips = ip_data.signal
vps = vp_data.signal
tb = ip_data.timebase
w = np.where(ck_data.signal > 0.15)

ends = w[0][np.where(np.diff(w[0]) > 2)[0]]
print('clock period = {n} samples'.format(n=ends[2] - ends[1]))
starts = ends[1:-1] + (ends[1] - ends[0]) // 4  # take the second half,
ends = ends[2:]

ip_means = [np.mean(ips[starts[i]:ends[i]]) for i, st in enumerate(starts)]
vp_means = [np.mean(vps[starts[i]:ends[i]]) for i, st in enumerate(starts)]
tb_means = [np.mean(tb[starts[i]:ends[i]]) for i, st in enumerate(starts)]
axt.plot(tb_means, -1 * np.array(ip_means), 'or', label='mean(-ip): c.f. isat')
axt.plot(tb_means, np.array(vp_means)/200, '.g', label='mean(vp(t))/200')

if debug > 0:
    axt.plot(tb[w], -ips[w], '.m', label='-ip(t) during clock')
    for i, st in enumerate(starts):
        axt.plot(tb[starts[i]:ends[i]], -ips[starts[i]:ends[i]], '.r',
                 label=['-ip used','_'][i>0])
num_sets = len(ip_means) // 3
ip_grouped = np.reshape(ip_means[0:num_sets * 3], [num_sets, 3])
sat_set = np.argmax(-np.sum(ip_grouped, axis=0))
# could put in order here so i_ion was first (grouped[:,0]) but no need
n_used = num_sets - 1
ip_grouped = np.reshape(ip_means[sat_set:sat_set + n_used * 3], [n_used, 3])
vp_grouped = np.reshape(vp_means[sat_set:sat_set + n_used * 3], [n_used, 3])
tb_grouped = np.reshape(tb_means[sat_set:sat_set + n_used * 3], [n_used, 3])

fits = []
plots = 0
for tbg, ip, vp in zip(tb_grouped, ip_grouped, vp_grouped):
    guess = vp[-1] - vp[0], np.mean(vp), np.mean(np.abs(ip))
    (Te, Vf, i0), flag = leastsq(residuals, guess, args=(ip, vp))
    if debug > 2:
        print('{f}: dV = {dV:.2f}, Te={Te:.2f}, Vf={Vf:.2f}, i0={i0:.3f},'  # offs={offs:.2f}, i_imbalance={im:.1f}%'
              .format(f=flag, dV=vp[0] - vp[-1], Te=Te, Vf=Vf, i0=i0))  # offs=offs, im = 100*i_p.mean()/np.max(np.abs(i_p))))
    fits.append([tbg[0], Te, Vf, i0])
    if plots < numplots:
        if plots == 0:
            fig,axc = plt.subplots(1,1)
        plots += 1
        vv = np.linspace(-90, 50, 1000)
        axc.plot(vp, ip, 'o')  #, label='After {n} iters'.format(n=nits))
        print(Te, Vf, i0)
        print(ip)
        axc.plot(vv, LPchar(vv, Te, Vf, i0), label='fitted LP char')
        axc.legend(loc='best')
        plt.show()

fits = np.array(fits, dtype=np.float32)
wnan = np.where((fits[:,-1] > 0) | (fits[:,-1] < -1))[0]
fits[wnan,-1] = np.nan
axt.step(fits[:, 0], -fits[:, -1], where='mid',label='isat_fitted')
axt.set_ylim(0, axt.get_ylim()[1])
axt.legend(loc='best')
