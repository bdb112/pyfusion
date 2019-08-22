from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
global lastxy, plot_slope, connect_id

def onclick(event):
    global lastxy, plot_slope, connect_id
    print(event.xdata, event.ydata, event.button)  # print at full res once
    ax = event.inaxes
    if event.button != 2:
        print('skipping slope - use middle button for slope', end=': ')
    if event.button == 2 and lastxy is not None and lastxy[0] is not None and lastxy[0] != event.xdata:
        dx = (event.xdata - lastxy[0])
        dy = (event.ydata - lastxy[1])
        print(' dx = {dx:.4g}, dy = {dy:.4g}'.format(dx=dx, dy=dy))
        if dx != 0:
            dy = dy if ax.get_yscale() == 'linear' else (np.log(event.ydata) - np.log(lastxy[1]))
            dx = dx if ax.get_xscale() == 'linear' else (np.log(event.xdata) - np.log(lastxy[0]))
            # print('dx, dy', dx, dy,  (np.log(event.xdata), np.log(lastxy[0])))
            sl = dy / dx
            print("1/t = {fr:.4g}, slope, inv = {sl:.4g}, {inv:.4g}, average = {av:.4g}"
                  .format(fr=1/dx, sl=sl, inv=1/sl, av=(event.ydata + lastxy[1])/2.0))
        if plot_slope:
            print('plotting slope', end=': ')
            x = np.linspace(lastxy[0] - dx, event.xdata + dx)
            ax.plot(x, event.ydata + sl * (x - event.xdata), '--')
            ax.plot([lastxy[0], event.xdata], [lastxy[1], event.ydata], lw=4, color=ax.get_lines()[-1].get_color())
            plt.show()
            lastxy = None  # prevent another false slope - wait for 2 points
    else:
        lastxy = event.xdata, event.ydata


def remove():
    fig.canvas.mpl_disconnect(connect_id)
    plot_slope = True
    
# main
plot_slope = True
lastxy = None
fig = plt.gcf()
connect_id = fig.canvas.mpl_connect('button_press_event', onclick)
print('click first with left button the with middle button for slope', 'remove() to remove event handler' )


"""
run pyfusion/examples/process_MLP.py shot_number=[20181018,34] double=1 time_range=[1.0000012,1.001] mode='Te' debug=0
i=boxcar(ip_data.signal, 291.225/3, maxnum=4)
v=boxcar(vp_data.signal, 291.225/3, maxnum=4)
t=boxcar(ip_data.timebase, 291.225/3, maxnum=4)
fig,axs = plt.subplots(2, 1, sharex='all')
axs[0].plot(t, v)
axs[1].plot(t, i)
run ~/python/cdiff.py
"""
