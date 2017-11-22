""" Version based on matplotlib example.  
Add title, choice of scatter (slow, but shows Amp and a12) or (fast) plot

"""

from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np

ax = subplot(111)
subplots_adjust(left=0.15, bottom=0.25, right=0.99)
scale_0 = 5
debug = 0
global N, pmode

try:
    dd
    print('using memory copy')
except:
    dd=np.load('MP512all.npz')['dd'].tolist()

#create_widget()
shots = np.unique(dd['shot'])

axcolor = 'lightgoldenrodyellow'

ax_shot  = axes([0.15, 0.15, 0.65, 0.03], axisbg=axcolor)
ax_scale = axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)

sl_shot = Slider(ax_shot, 'Shot Index', 0, len(shots), valinit=0, valfmt='%d')
sl_scale = Slider(ax_scale, 'Scale', np.log(0.01), np.log(300), valinit=np.log(scale_0), valfmt="%.1f")

resetax = axes([0.8, 0.025, 0.13, 0.04])
button = Button(resetax, 'Reset Axes', color=axcolor, hovercolor='0.975')

def minmax(list): return(min(list), max(list))

def reset(event):
    # axrelim() doesn't apply to collections? yet?
    if len(l.get_xdata()) == 0: return
    ax.set_xlim(minmax(l.get_xdata()))
    ax.set_ylim(minmax(l.get_ydata()))
    clim(0,1)
    draw()

button.on_clicked(reset)

plotmodeax = axes([0.025, 0.4, 0.08, 0.09], axisbg=axcolor)
plotmoderadio = RadioButtons(plotmodeax, ("Fast", "Cols"), active=1)

pmode='Cols'
def plotmodefunc(label):
    global pmode
    pmode = label
    draw()
plotmoderadio.on_clicked(plotmodefunc)

rax = axes([0.025, 0.5, 0.08, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ("N=0", "N=1", "N=2"), active=2)

N = 2
def NFunc(label):
    global N
    N = int(label.split('=')[-1])
    draw()
radio.on_clicked(NFunc)

def get_data(shotind, scale=5):
    if 'N' in dd: 
        w = where((shots[shotind] == dd['shot']) & (dd['N']==N))[0]
    else:
        w = where((shots[shotind] == dd['shot']))[0]
    if debug: print(shots[shotind], w)
    x = dd['t_mid'][w] 
    y = dd['freq'][w] 
    s = 10*np.log((dd['amp'][w])/scale)
    c = dd['a12'][w]
    ax.set_title(shots[shotind])
    return (x,y,s,c)

shotind = sl_shot.val
(x,y,siz,col) = get_data(shotind)

axes(ax)
l, = plot([],[],marker='o', linestyle='') # can't use array for marker size
#axis([0, 1, -10, 10])
splot = scatter(x,y,siz,col,marker='o',vmin=0, vmax=1)
colorbar()

def update(val):
    siz_scale = np.exp(sl_scale.val)
    shotind = sl_shot.val
    sl_scale.valfmt = "%0d {:.2g}".format(siz_scale)
    (x,y,siz,col) = get_data(shotind, siz_scale)
    if pmode == 'Fast':
        l.set_xdata(x)
        l.set_ydata(y)
        if len(splot.get_offsets().flatten()) != 0:
            splot.set_offsets(array([[],[]]))
            splot.set_array(array([]))
            splot._sizes=array([])
    else:
        if len(l.get_xdata())>0:
            l.set_xdata([])
            l.set_ydata([])
        splot.set_offsets(np.array([x,y]).T)
        splot.set_array(col)
        splot._sizes = siz
    draw()

sl_shot.on_changed(update)
sl_scale.on_changed(update)

show(0)
