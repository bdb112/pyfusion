import pyfusion
import matplotlib.pyplot as plt

from pyfusion.utils import process_cmd_line_args

_var_defaults = """
shot_list = [[20160310,i] for i in range(7,40)]
diag = 'W7X_TotECH'
dev_name = 'W7X'
"""
exec(_var_defaults)
exec(process_cmd_line_args())

dev = pyfusion.getDevice(dev_name)

i = 0
axs = []
for shot in shot_list:
    if i>=len(axs):
        fig, axs = plt.subplots(nrows=3, ncols=4, squeeze=False, figsize=[12,8])
        axs = axs.flatten()
        fig.subplots_adjust(left=0.08, bottom=0.07, right=0.96, top=0.90, wspace=.35, hspace=0.2)
        i = 0

    plt.sca(axs[i])
    try:
        dev.acq.getdata(shot, diag).plot_signals(suptitle='')
        axs[i].set_title(str(shot))
        i += 1
    except LookupError:
        pass

plt.show()
