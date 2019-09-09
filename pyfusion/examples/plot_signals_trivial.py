import pyfusion as pf
dev = pf.getDevice("W7X")
data = dev.acq.getdata([20181009, 24], 'W7X_MIRNOV_41_3')
data.plot_signals()
