"""
Note, plots (this file) doesn't have unittests

Problems with checkbuttons working since 2011-2012? 
Temporary fix is to block in svdplot when hold==1.
This enables the function, then you kill the window to move on (see plot_svd.py)
Also attempted to use subplots here to tidy up putting additonal graphs on top.
See plots_1.py and svd_plots1.py - need to sovle subplot pars problem
bdb made format more conforming. 2013
"""
from __future__ import division
from matplotlib.widgets import CheckButtons
from matplotlib.font_manager import FontProperties
import pylab as pl

import numpy as np

from pyfusion.data.utils import split_names, make_title, get_axes_pixcells #, peak_freq
from pyfusion.data.filters import cps
from pyfusion.utils.utils import fix2pi_skips, modtwopi
from pyfusion.debug_ import debug_

#can't from pyfusion.config import get
import pyfusion

def overlap(interval1, interval2):
    """ return true if there is any overlap between the two intervals 
    probably belongs in convenience or utils
    """
    return not (min(interval1)>max(interval2) 
                or (max(interval1)<min(interval2)))

def set_axis_if_OK(ax, xlims, ylims):
    """
    set axes if they containg at least a bit of the plot
    originally meant to be used when forcing specific axes, to allow
    for attempt to force axes suitable for LHD on HJ data for example
    """
    from pyfusion.utils.utils import warn
    if overlap(ax.get_xlim(), xlims) and overlap(ax.get_ylim(), ylims):
        ax.axis(xlims[0], xlims[1], ylims[0], ylims[1])
    else:
        warn('suppressed attempt to set axis outside ranges\n{att:s} cf {act:s}'
             .format(att=(xlims,ylims), act=(ax.get_xlim(),ax.get_ylim())))



plot_reg = {}

# the registration function is similar but separate for plots and filters
def register(*class_names):
    def reg_item(plot_method):
        for cl_name in class_names:
            if cl_name not in plot_reg:
                plot_reg[cl_name] = [plot_method]
            else:
                plot_reg[cl_name].append(plot_method)
        # See Note in base.py line 76 about when this works.
        debug_(pyfusion.DEBUG, 4, key=['register plot', 'register'])
        return plot_method
    return reg_item

def mydiff(t,y):
    """ numpy diff in a wrapper to give correct time and units"""
    return((t[0:-1] + t[1:])/2, np.diff(y)/np.average(np.diff(t)))

def myiden(y):
    return(y)

def myiden2(t,y):
    return(t,y)

@register("TimeseriesData")
def plot_signals(input_data, filename=None, downsamplefactor=1, n_columns=1, hspace=None, sharey=False, sharex=True, ylim=None, xlim=None, marker='None', markersize=2, linestyle=True, labelfmt="{short_name} {units}", filldown=True, suptitle='shot {shot}',raw_names=False, labeleg='False',color='b', t0=0, fun=myiden, scale=1, offset=0, fun2=None, **kwargs):
    """ 
    Plot a figure full of signals using n_columns[1], or just one signal
        sharey [=1 or 'all']  "gangs" y axes  - sim. for sharex - sharex=None stops this
        sharey: 2 gangs all but first (top) axis
        x axes are ganged by default: see Note:

    downsamplefactor = 1 # 10 plots 1/10th of the data.
    color
    marker
    linestyle
    markersize
    n_columns = 1
    scale = 1  # only works if fun2 is None
    offset = 0  # plots offset + scale * sig
    fun, fun2: optionally plot a function of the signal.  fun= refers
    to a function of one variable.  fun2 is a function (t,x), such as
    one that returns a different timebase (diff should do this)
    if fun2 is given (a function of t and sig), then fun is ignored 


        labelfmt["{short_name} {units}"] controls the channel labels.  
            The default ignores the shot and uses an abbreviated form of the channel name.  
            If the short form is very short, it becomes a y label.
            A full version is "{name} {units}" and if > 8 chars, will become the x label.
            Even longer is "Shot={shot}, k_h={kh}, {name}"
        labeleg: If 'True' put label in legend - else use this str as a legend lab
        linestyle: default of True means use '-' if no marker, and nothing if a 
            marker is given      
            e.g. marker of '.' and markersize<1 will produce "shaded" waveforms, 
            good to see harmonic structure even without zooming in (need to 
            adjust markersize or plot size for best results.
        raw_names uses "digitiser" names, otherwise use names from the config file
        Note = sharex that to allow implicit overlay by using the same subplot
        specs the sharex must be the same between main and overlay - hence the 
        use of explicit sharex = None
        
        suptitle by default refers to the shot number
    """
    if isinstance(sharey, str) and sharey.lower() == 'all':
        sharey = 1
    elif isinstance(sharey, type(max)):  # really want isinstance(sharey, function) but not defined
        raise ValueError('''sharey was set to the function all()! - use sharey=1 or "sharey='all'"''')

    if pyfusion.NSAMPLES > 0 and downsamplefactor != 1:
        print('Ignoring downsample on already decimated signal')
        downsamplefactor = 1
        
    if pyfusion.VERBOSE > 1: print(fun, fun2)
    if ((scale != 1) or (offset != 0)) and fun2 is not None:
        raise ValueError('need fun2 to be None so that scale/offset works')
    import pylab as pl
    n_rows = input_data.signal.n_channels() # doesn't work with fftd data
    n_rows = int(round(0.49+(n_rows/float(n_columns))))
    if (n_rows > 3) and (hspace is None): 
        hspace = 0.001 # should be 0, but some plots omitted if 
                       #exactly zero - fixed in matplotlib 1
    if pyfusion.VERBOSE > 3: print(str(n_rows) + ' ' + str(n_columns))

    if labelfmt != None:
        if len(make_title(labelfmt, input_data, 0, raw_names=raw_names)) > 11: 
            mylabel = pl.xlabel
        else:
            mylabel = pl.ylabel
    fontkwargs = {'fontsize': 'small'}        
    # True is the only sensible indicator I can think of that we want intelligient defaults
    # If linestyle == True, we default to '-' UNLESS marker is set, in which case default to '' (no line)
    if linestyle == True:
        if marker == 'None': linestyle = '-'
        else: linestyle = ''
        
    axcount = -1  # so the first will be 0
    for row in range(n_rows):
        for col in range(n_columns):
            axcount += 1
            # natural sequence for subplot is to fillacross l-r, then top-down 
            subplot_num = row*n_columns+col

            # we often want to fill downwards for simple arrays - especially if comparing with a 3x16 array
            if filldown: chan_num = col*n_rows+row
            else:        chan_num = row*n_columns+col

            if chan_num >= input_data.signal.n_channels(): break
            if pyfusion.VERBOSE>3: print(subplot_num+1,chan_num)
            if axcount == 0:
                # note - sharex=None is required so that overlays can be done
                if n_rows * n_columns == 1:
                    axlead = pl.gca()  # this allows plotting on existing axis for a single plot
                else:
                    axlead = pl.subplot(n_rows, n_columns, subplot_num+1, sharex = None)
                axn = axlead
                axlead_x = axlead if sharex else None
            else:
                if axcount >= sharey: 
                    axn = pl.subplot(n_rows, n_columns, subplot_num+1, sharex = axlead_x, sharey=axlead)
                else: # another noshare y, but sharex
                    axn = pl.subplot(n_rows, n_columns, subplot_num+1,
                                     sharex = axlead_x)
                    axlead = axn

            #  Clumsy addition to put labels in the legend
            #  To do better, should be one or the other, but the
            #  original code is after the plot - need to move it
            # back to here, before the plot
            ##### No! should use text(0.5,1,'foo',verticalalignment='top',transform=gca().transAxes)
            
                
            if labeleg == 'False':
                lab = ''
            elif labeleg == 'True':
                sht = str(input_data.meta['shot'])
                lab = make_title(sht + ' ' + labelfmt, input_data, chan_num)
            else:
                lab = labeleg

            KWargs = dict(marker=marker, markersize=markersize, 
                          linestyle=linestyle, label = lab, color=color)
            KWargs.update(kwargs)

            if fun2 is not None:  # can't easily accommodate 'scale/offset' here
                pl.plot(*fun2(input_data.timebase[::downsamplefactor]-t0, 
                              input_data.signal.get_channel(
                                  chan_num)[::downsamplefactor]),**KWargs)
            else:
                pl.plot(input_data.timebase[::downsamplefactor]-t0, 
                        offset + scale * fun(input_data.signal.get_channel(
                            chan_num)[::downsamplefactor]),**KWargs)

# this old code was no faster
#           #   if downsamplefactor==1:
#            pl.plot(input_data.timebase, 
#                    input_data.signal.get_channel(chan_num),
#                    **kwargs)
#            else:
#                plotdata=input_data.signal.get_channel(chan_num)
#                timedata=input_data.timebase
#                pl.plot(timedata[0:len(timedata):downsamplefactor], 
#                        plotdata[0:len(timedata):downsamplefactor], 
#                        **kwargs)
#                pl.axis([-0.01,0.1,-5,5])

            pl.xticks(**fontkwargs)
            pl.yticks(**fontkwargs)

            if labelfmt != None:    
                if mylabel == pl.ylabel and np.mod(row,2): displace='\n'
                else: displace = ''  # use \n to make two line label to displace every second one

                lab = make_title(labelfmt+displace, input_data, chan_num)
                mylabel(lab,**fontkwargs)

            if n_rows>3: 
                #print('locator_params',int(25/np.sqrt(n_rows)))
                axn.locator_params(prune='both', axis = 'y',
                                   nbins=min(5,int(25/np.sqrt(n_rows))))
                # this gets rid of the x labels at either end until we 
                # can suppress xlabels on all but the lowest

                if n_rows>1: axn.locator_params(prune='both', axis = 'x')

            if ylim != None: pl.ylim(ylim)
            if xlim != None: pl.xlim(xlim)
    if suptitle is not None:
        try:
            suptitlestr = (suptitle.format(**input_data.meta))
        except:
            suptitlestr = ''
            debug_(pyfusion.DEBUG, 1, key='plot_signal_suptitle',
                   msg=' input metadata [{m}] does not have  a '
                   'key for suptitle [{s}]'
                   .format(m=input_data.meta, s=suptitle))

    if hspace is not None: # adjust vertical spacing between plots
        pl.gcf().subplotpars.hspace = hspace
        pl.gcf().subplotpars.bottom = hspace + 0.08 # was 0.04
        extratop = 0.01
        if suptitlestr != '': extratop += 0.04
        pl.gcf().subplots_adjust(top = 1-(hspace+extratop)) # allow a little room for title

    # suptitle can have references to show number etc in metadata
    if suptitlestr != '': pl.suptitle(suptitlestr)
    if filename != None:
        pl.savefig(filename)
    else:
        pl.show(block=0)
    debug_(pyfusion.DEBUG, 2, key='plot_signals')

@register("TimeseriesData")
def plot_spectrogram(input_data, windowfn=None, units='kHz', channel_number=0, filename=None, coloraxis='now is clim!', clim=None, xlim=None, ylim=None, noverlap=0,NFFT=None, suptitle='shot {shot}', title=None, sharey=True, sharex=True, n_columns=None, raw_names=False, hspace=None, labelfmt="{short_name} {units}", filldown=False,hold=None,**kwargs):
    """    Plot a spectrogram 
      NFFTs
      noverlap  (integer - number of samples overlapped, float - 1.0 -> half nFFT)
      windowfn (p.window.hanning)
      coloraxis - gets from pyfusion.conf.get('Plots')

    Accepts multi or single channel data (I think?)
    Title will be auto generated: if supplied, include '+' to include the auto-generated part
    To suppress, use title=' ' (one space)
    clim:  [None] auto clim, 'show' - label each graph with clims
    Returns ax_list which can be used for clims or other manipulations
    """
    import pylab as pl
    
    # can't recurse as this is a signal   input_data[chan.name].plot_specgram()

    if hold is not None and hold == 0:
        pl.figure()
    if windowfn is None: windowfn=pl.window_hanning

    # look in the config file section Plots for NFFT = 1234
    # Dave - how about a method to allow this in one line
    # e.g. pyfusion.config.numgetdef('Plots','NFFT', 2048)
    # usage:  
    # if (NFFT is None): NFFT = pyfusion.config.numgetdef('Plots','NFFT', 2048)
    # 
    # also nice to have pyfusion.config.re-read()
    if NFFT is None:
        try:
            NFFT=(int(pyfusion.config.get('Plots','NFFT')))
        except:
            NFFT = 2048

    print(NFFT)        
    if units.lower() == 'khz': ffact = 1000.
    else: ffact =1.        
    xextent=(min(input_data.timebase),max(input_data.timebase))


    n_pics = input_data.signal.n_channels() # doesn't work with fftd data
    if n_columns is None:
        n_columns = int(0.8 + np.sqrt(n_pics))
    n_rows = int(round(0.49+(n_columns/float(n_pics))))
    while n_rows * n_columns < n_pics:
        n_rows += 1
        
    if (n_rows > 3) and (hspace is None): 
        hspace = 0.001 # should be 0, but some plots omitted if 
                       #exactly zero - fixed in matplotlib 1
    if pyfusion.VERBOSE > 3: print(str(n_rows) + ' ' + str(n_columns))

    fontkwargs = {'fontsize': 'small'}        
    # True is the only sensible indicator I can think of that we want intelligient defaults
    displace = '' # doens't make send for spectra, as they are usually squarish
    
    axcount = -1  # so the first will be 0
    ax_list = []  #  We don't use subplots(), because we want control of sharey
    for row in range(n_rows):
        for col in range(n_columns):
            axcount += 1
            # natural sequence for subplot is to fillacross l-r, then top-down 
            subplot_num = row*n_columns+col

            # we often want to fill downwards for simple arrays - especially if comparing with a 3x16 array
            if filldown: chan_num = col*n_rows+row
            else:        chan_num = row*n_columns+col

            #print(chan_num, subplot_num, col, row)
            if chan_num >= input_data.signal.n_channels(): break
            if pyfusion.VERBOSE>3: print(subplot_num+1,chan_num)

            if pyfusion.VERBOSE>3: print(subplot_num+1,chan_num)
            if axcount == 0:
                # note - sharex=None is required so that overlays can be done
                if n_rows * n_columns == 1:
                    axlead = pl.gca()  # this allows plotting on existing axis for a single plot
                else:
                    axlead = pl.subplot(n_rows, n_columns, subplot_num+1, sharex = None)
                axn = axlead
                axlead_x = axlead if sharex else None
            else:
                if axcount >= sharey: 
                    axn = pl.subplot(n_rows, n_columns, subplot_num+1, sharex = axlead_x, sharey=axlead)
                else: # another noshare y, but sharex
                    axn = pl.subplot(n_rows, n_columns, subplot_num+1,
                                     sharex = axlead_x)
                    axlead = axn
    
            noverlap = noverlap if isinstance(noverlap, int) else int(round(NFFT/(1 + 1./(1e-6 + noverlap))))
                
            (specarr, freqs, t, im) = \
                axn.specgram(input_data.signal.get_channel(chan_num),
                             NFFT=NFFT, noverlap=noverlap,
                             Fs=input_data.timebase.sample_freq/ffact,
                             window=windowfn, xextent=xextent, **kwargs)
            ax_list.append(axn)
            # Used be (incorrectly coloraxis)
            if pyfusion.VERBOSE>2:
                print('data/plot_spectrogram: noverlap={no}, {nt} time segs, {nf} freqs'
                      .format(no=noverlap, nt=len(t), nf=len(freqs)))
            if xlim is not None:
                axn.set_xlim(xlim)
            if ylim is not None:
                axn.set_ylim(ylim)
            if clim is not None and clim != 'show': im.set_clim(clim)
            else:
                try:
                    pl.clim(eval(pyfusion.config.get('Plots','coloraxis')))
                except:
                    pass

            if labelfmt != None:
                if len(make_title(labelfmt, input_data, 0, raw_names=raw_names)) > 11: 
                    mylabel = pl.xlabel
                else:
                    mylabel = pl.ylabel

            lab = make_title(labelfmt+displace, input_data, chan_num)
            mylabel(lab,**fontkwargs)

            # look in the config file section Plots for a string like 
            # FT_Axis = [0,0.08,0,500e3]   don't quote
            exceptions_to_hide = Exception if pyfusion.DBG() < 3 else None
            try:
                #pl.axis(eval(pyfusion.config.get('Plots','FT_Axis')))
                # this is clumsier now we need to consider freq units.
                axt = eval(pyfusion.config.get('Plots','FT_Axis'))
                set_axis_if_OK(pl.gca(),axt[0:2], np.array(axt[2:])/ffact)
            except exceptions_to_hide:
                pass
            # but override X if we have zoomed in bdb
            if 'reduce_time' in input_data.history:
                pl.xlim(np.min(input_data.timebase),max(input_data.timebase))


            try:
                tit = str("{s}, {c}"
                          .format(s=input_data.meta['shot'], c=input_data.channels[chan_num].config_name))
            except:
                tit = str("{s}, {c}"
                          .format(s=input_data.meta['shot'], c=input_data.channels.name))
            if title is None or title == '':  # get the default title
                pass # tit is the default
            else:
                tit = title.replace('+',tit)
            # No titles for rows >= 3 - maybe better to have no titles at all
            #   and use the legend (without frame) to show the channel number
            # instead of the ytitle.
            if n_rows <= 3:  # keep consistent with n_rows > 3 above
                pl.title(tit)
            if clim == 'show':
                axn.legend([], [], frameon=0, title=str(np.round(axn.get_images()[0].get_clim(), 1)))

    # ======== end of plot loop
    if suptitle is not None:
        try:
            suptitlestr = (suptitle.format(**input_data.meta))
        except:
            suptitlestr = ''
            debug_(pyfusion.DEBUG, 1, key='plot_signal_suptitle',
                   msg=' input metadata [{m}] does not have  a '
                   'key for suptitle [{s}]'
                   .format(m=input_data.meta, s=suptitle))


    if suptitle != '': pl.suptitle(suptitlestr)
    if hspace is not None: # adjust vertical spacing between plots
        pl.gcf().subplotpars.hspace = hspace
        pl.gcf().subplotpars.bottom = hspace + 0.08 # was 0.04
        extratop = 0.01
        if suptitle != '': extratop += 0.04
        pl.gcf().subplots_adjust(top = 1-(hspace+extratop)) # allow a little room for title

    if filename != None:
        pl.savefig(filename)
    else:
        pl.show(block=0)

    return(ax_list)

#def plot_multichannel_coord(input_data, coord=None, savefig=None):
#    pass


def join_ends(inarray,add_2pi = False,add_360deg=False,add_lenarray=False,add_one=False):
    """used in old code, needs clean up...."""
    output = np.resize(inarray,(len(inarray)+1,))
    if add_2pi:
        output[-1] = output[-1]+2*pi
    elif add_360deg:
        output[-1] = output[-1]+360.0
    elif add_lenarray:
        output[-1] = output[-1]+len(inarray)+1
    elif add_one:
        output[-1] = output[-2]+1
    return output

def posNegFill(x,y1,y2):
    diff = y2 - y1
    pos = []
    neg = []
    xx1 = [x[0]]
    xx2 = [x[0]]
    yy1 = [y1[0]]
    yy2 = [y2[0]]
    oldSign = (diff[0] < 0 )
    npts = x.shape[0]
    for i in range(1,npts):
        newSign = (diff[i] < 0)
        if newSign != oldSign:
            xz,yz = findZero(i,x,y1,y2)
            xx1.append(xz)
            yy1.append(yz)
            xx2.reverse()
            xx1.extend(xx2)
            yy2.reverse()
            yy1.extend(yy2)
            if oldSign:
                neg.append( (xx1,yy1) )
            else:
                pos.append( (xx1,yy1) )
            xx1 = [xz,x[i]]
            xx2 = [xz,x[i]]
            yy1 = [yz,y1[i]]
            yy2 = [yz,y2[i]]
            oldSign = newSign
        else:
            xx1.append( x[i])
            xx2.append( x[i])
            yy1.append(y1[i])
            yy2.append(y2[i])
            if i == npts-1:
                xx2.reverse()
                xx1.extend(xx2)
                yy2.reverse()
                yy1.extend(yy2)
                if oldSign :
                    neg.append( (xx1,yy1) )
                else:
                    pos.append( (xx1,yy1) )
    return pos,neg

class Energy:
    def __init__(self,energy_list,initial_list):
        self.value = 0.0
        self.energy_list = energy_list
        for i in range(len(initial_list)):
            if initial_list[i]:
                self.value += self.energy_list[i]

    def add(self,elmt):
        self.value += self.energy_list[elmt]

    def sub(self,elmt):
        self.value -= self.energy_list[elmt]

def findZero(i,x,y1,y2):
    im1 = i-1
    #print y1[i],y1[im1],x[i],x[im1],(x[i]-x[im1])
    m1 = (y1[i] - y1[im1])/(x[i] - x[im1])
    m2 = (y2[i] - y2[im1])/(x[i] - x[im1])
    b1 = y1[im1] - m1*x[im1]
    b2 = y2[im1] - m2*x[im1]
    xZero = (b1 - b2)/(m2 - m1)
    yZero = m1*xZero + b1
    return (xZero, yZero)

@register("FlucStruc")
def fsplot_phase(input_data, closed=True, ax=None, hold=0, offset=0, AngName=None, block=False):
    """ plot the phase of a flucstruc, optionally inserting the first point
    at the end (if closed=True). Applies to closed arrays (e.g complete 2pi).
    Until Feb 2013, this version did not yet attempt to take into account angles, or check 
    that adjacent channels are adjacent (i.e. ch2-ch1, ch2-c2 etc).
    Channel names are taken from the fs and plotted abbreviated

    1/1/2011: TODO This appears to work only for database = None config
    1/17/2011:  bdb: May be fixed - I had used channel instead of channel.name
    """
    # extract by channels
    ch1n,ch2n,ch21n,dp = [],[],[],[]
    # bdb this line should be replaced by a call to a routine names something
    #like <plotted_width> to help in deciding if the label will fit on the 
    #current graph.
    if ax is None:     ax=pl.gca()
    # why is this repeated below
    max_chars = get_axes_pixcells(ax)[2]/8 # assuming an 10 pt font is 8 wide.
    if (2*len(input_data.dphase)
        *(2+len(input_data.dphase[0].channel_1.name)))> max_chars:
        sep = '\n-'
    else: sep = '-'
    
    #sep = '-'
    # 2013 change order from ch1-ch2 to ch2-ch1 = ch2 - ch1
    for dpn in input_data.dphase:
        ch1n.append(dpn.channel_1.name)
        ch2n.append(dpn.channel_2.name)
        ch21n.append(dpn.channel_2.name+sep+dpn.channel_1.name)
        dp.append(dpn.delta)

    min_length = max(1,40/len(input_data.channels))    
    # min_length - min_length???
    short_names_1,p,s = split_names(ch1n, min_length=2)  # need to break up loops to do this
    short_names_2,p,s = split_names(ch2n, min_length=5)  # 

# need to know how big the shortened names are before deciding on the separator
    if (2*len(input_data.dphase)*(2+len(short_names_1[0])))> max_chars:
        sep = '\n-'
    else: sep = '-'

    ch21n = [ch2n[i]+sep+ch1n[i] for i in range(len(ch1n))]
    short_ch21n = [short_names_2[i]+sep+short_names_1[i] 
                   for i in range(len(short_names_1))]

    # correct result for a diag that doesn MP1,2,3,4,5,6,1, and not closed
    #ch21n = ['MP2-MP1', 'MP3-MP2', 'MP4-MP3', 'MP5-MP4', 'MP6-MP5', 'MP1-MP6']
    # if closed, and only 1-6 in the file, want the same!

    if closed:   # ch2 will be MP1, ch1 will be MP6   MP1-MP6
        ch1n.append(ch2n[-1])
        ch2n.append(ch1n[0])
        ch21n.append(ch2n[-1]+sep+ch1n[0])
        short_ch21n.append(short_names_2[-1]+sep+short_names_1[0])
        #dp.insert(0,dp[-1])
        # closed means use the phase diff from the ends
        # sign bug fixed here - made closed opposite sign to open..
        dp=np.append(dp,modtwopi(-np.sum(dp),offset=offset))

    #dp = fix2pi_skips(dp, around=offset)
    dp = modtwopi(dp, offset=offset)

    if hold == 0: ax.clear()

    # This is a kludgey way to read coordinates.  Should be through acquisition.base or acquisition.'device'
    Phi = np.array([2*np.pi/360*float(pyfusion.config.get
                                      ('Diagnostic:{cn}'.
                                       format(cn=c.config_name), 
                                       'Coords_reduced')
                                      .split(',')[0]) 
                    for c in input_data.channels])
    Theta = np.array([2*np.pi/360*float(pyfusion.config.get
                                        ('Diagnostic:{cn}'.
                                         format(cn=c.config_name), 
                                         'Coords_reduced')
                                        .split(',')[1]) 
                      for c in input_data.channels])

    if closed: 
        Phi = np.append(Phi, Phi[0])
        Theta = np.append(Theta, Theta[0])

    # kludgey - needed to deal with Shaun's array - should
    # really convert form cylindrical
    if AngName is not 'Phi' and len(np.unique(Theta)) > 1:
        AngName = 'Theta'
        Ang = Theta
        dAngfor1 = np.pi/13  #dAngfor1 is the average coil dph for M=1
        span = (np.pi)/13  # span is the average coil spacing in radians
    else:
        AngName = 'Phi'
        Ang = Phi
        dAngfor1 = (2*np.pi)/6
        span = 2*np.pi/6
        
    # expect Phi from ~.2 to twopi+.2 (if closed)
    Angfix = fix2pi_skips(Ang,around=3.5)

    # Here use fix2pi_skips, as we want the coil location angle to 
    # increase monotonically - the raw numbers in .cfg may not start at zero 
    # need to make sure the right dp is divided by the right dPhi!
    ax.plot(fix2pi_skips(Angfix[0:-1],around=np.pi),dp/np.diff(Angfix)*span,
            '+-',label='dp/d'+AngName[0])
    ax.plot(fix2pi_skips(Angfix[0:-1],around=np.pi),dp,'o-', label='d'+AngName)
    ax.set_xlim([ax.get_xlim()[0],
                 ax.get_xlim()[1]+np.average(np.diff(Angfix))])
    ax.plot(ax.get_xlim(),[0,0],':k',linewidth=0.5)
    for N in (-3,-2,-1,1,2,3): ax.plot(ax.get_xlim(),[N*dAngfor1,N*dAngfor1],':r',linewidth=0.5)
    tot=np.sum(dp)
    over = Angfix[-1] - Angfix[0]
    print(tot)
    if closed: # only makes sense to draw a mean dp line if closed
        ax.plot(ax.get_xlim(),[tot/(2*np.pi),tot/(2*np.pi)],':b',linewidth=0.5)

    ax.legend(prop=FontProperties(size='small'))
    ax.set_title('sum = {s:.2f} over {o:.2f} ~ {r:.1f}'
                 .format(s=tot,o=over, r=tot/over))
    debug_(pyfusion.DEBUG, 1, key='fs_phase')
    #ax.set_xticks(range(len(dp)))
    ax.set_xticks(Angfix)
    ax.set_xticklabels(short_ch21n)
    pl.show(block=block)

@register("SVDData")
def svdplot(input_data, fmax=None, hold=0, init_but_list=[0,1]):

    if hold==0: pl.clf() # erase the figure, as this is a multi-axis plot
    else: pl.clf() # do it anyway, trying to overcome checkbuttons problem

    n_SV = len(input_data.svs)

    #for chrono in input_data.chronos:
    #    print(peak_freq(chrono, input_data.dim1))

    # define axes 
    ax1 = pl.subplot(221)
    ax2 = pl.subplot(222)
    ax3 = pl.subplot(223)
    ax4 = pl.subplot(224)

    # allow space for check buttons
    pl.subplots_adjust(left=0.2, right=0.98)

    # setup check boxes
    rax = pl.axes([0.01, 0.05, 0.09, 0.9])

    # CheckButtons take tuple arguments, tuples are immutable, so create lists fom svd info, and then cast as tuple
    button_name_list=[]
    button_setting_list=[]

    for i in range(n_SV):
        button_name_list.append('  '+str(i))
        button_setting_list.append(False)
        
    # have first 2 SVs on as default
    for i in init_but_list:
        button_setting_list[i] = True

        # like "self"
    check = CheckButtons(rax, tuple(button_name_list), tuple(button_setting_list))
    # hack to make check buttons square
    check_box_stretch = 7
    for i in range(len(check.rectangles)):
    #if (1 == 0):  # use this line instead of for i in... to turn off the hack
        check.rectangles[i].set_width(check_box_stretch*check.rectangles[i].get_height())
        for j in [0,1]: # two lines of the x
            orig_x_data = check.lines[i][j].get_xdata()
            orig_y_data = check.lines[i][j].get_ydata()
            orig_width = orig_x_data[1]-orig_x_data[0]
            new_width = check_box_stretch*orig_width
            new_x_data = [orig_x_data[0],orig_x_data[0]+new_width]
            check.lines[i][j].set_data(new_x_data,orig_y_data)

    # plot all SVs, use button_setting_list for initial visibility
    # axes 1: chrono
    pl.axes(ax1)
    #pl.xlabel('Time -*-get units from Timebase-*-')
    pl.ylabel('chronos [a.u.]')  # used to be labelled amplitude, but is not really.
    # you can get to the reconstructed signals via fs.signal, but chronos is better here
    ## python3 - all these plot_lists refactored to look nicer in python2/3
    plot_list_1 = [ ax1.plot(np.arange(len(input_data.chronos[sv_i])), input_data.chronos[sv_i], visible= button_setting_list[sv_i],alpha=0.5)[0] for sv_i in range(n_SV)]
    #pl.xlim(min(input_data.dim1), max(input_data.dim1))

    # axes 2: SVs
    pl.axes(ax2)
    sv_sv = [input_data.svs[i] for i in range(n_SV)]
    ax2.semilogy(np.arange(n_SV),sv_sv,'ko',markersize=3)
    # avoid extreme ylim ranges - start with no more than 10^3
    (ymin,ymax) = ax2.get_ylim()
    ax2.set_ylim(max(ymin, ymax/1000), ymax)
    entropy = input_data.H
    pl.xlabel('Singular Value number')
    pl.ylabel('Singular Value')
#    pl.figtext(0.75,0.83,'1/H = %.2f' %(1./entropy),fontsize=12, color='r')
#    pl.figtext(0.75,0.81,'H = %.2f' %(entropy),fontsize=12, color='b')
    # this is done in two places - potential for inconsistency - wish I knew better -dgp
    # These changes make it easier to adjust the subplot layout
    # was pl.figtext(0.75,0.78, (relative to figure), make it relative to axes
# Use kwargs so that most formatting is common to all three labels.    
    kwargs={'fontsize':8,'transform':ax2.transAxes,
            'horizontalalignment':'right', 'verticalalignment':'top'}
    bsl = np.array(button_setting_list)
    RMS_scale=np.sqrt(np.mean(
            (input_data.scales[:,0]*bsl)**2))
    ### Note: amp is dodgy - calcualted roughly here
    ## reconsider how fs p is calculated!  maybe factor scales in before sum
    ## Note also - a12 is calcuated differently here  but looks OK
    # Also, shold update these like Dave does with Energy
    amp=np.sqrt(np.sum(input_data.p*bsl))*RMS_scale
    print("amp=%.3g:" % (amp)),
    energy = Energy(input_data.p,bsl)
    energy_label = ax2.text(0.96,0.98,'E = %.1f %%' %(100.*energy.value),
                            color='b', **kwargs)

    labstr = str('tmid = {tm:.1f} ms, 1/H = {invH:.2f}, below for 0,1, '\
                     'nrm={nrm}, ?Amp = {Amp:.2g}, a12 = {a12:.2f} '
                 .replace(', ','\n')
                 .format(tm=1e3*np.average(input_data.chrono_labels),
                         invH=(1./entropy), nrm = input_data.norm_method,
                         Amp = float(np.sqrt(np.sum(input_data.p*bsl))*RMS_scale),
                         a12 = float(np.sqrt(input_data.p[1]/input_data.p[0]))))
    ax2.text(0.96,0.85,labstr, color='r', **kwargs)

    # grid('True')
    plot_list_2 = []
    for sv_i in range(n_SV):        
        col = plot_list_1[sv_i].get_color()
        plot_list_2.append(ax2.semilogy([sv_i], [input_data.svs[sv_i]],
                                        '%so' %(col),visible= button_setting_list[sv_i],
                                        markersize=8,alpha=0.5)[0])

    # axes 3: fft(chrono)
    pl.axes(ax3)
    pl.xlabel('Frequency [kHz]')
    pl.ylabel('Power Spectrum')
    pl.grid(True)            # matplotlib 1.0.X wants a boolean (unquoted)
    nyquist_kHz = 1.e-3*0.5/np.average(np.diff(input_data.chrono_labels))
    plot_list_3 = []
    for sv_i in range(n_SV):
        col = plot_list_1[sv_i].get_color()
        tmp_chrono = input_data.chronos[sv_i]
        tmp_fft = np.fft.fft(tmp_chrono)[:len(tmp_chrono)//2]
        freq_array = nyquist_kHz*np.arange(len(tmp_fft))/(len(tmp_fft)-1)
        plot_list_3.append(ax3.plot(freq_array, abs(tmp_fft), col,visible= button_setting_list[sv_i],alpha=0.5)[0])
        
    if fmax is None: 
        ffact = 1e3  # seems like this routine is in kHz - where does it cvt?
        try:
            axt = eval(pyfusion.config.get('Plots','FT_Axis'))
            if axt[3]<2e3: ffact=1.0  # avoid a python3 issue 
            fmax = axt[3]/ffact
            fmin = axt[2]/ffact
        except:
            fmax = nyquist_kHz
            fmin = 0
    pl.xlim(fmin, fmax)

    # axes 4: topo
    pl.axes(ax4)
    pl.xlabel('Channel')
    pl.ylabel('Topo [a.u.]')
    angle_array = np.arange(n_SV+1)
    #channel_names = input_data.timesegment.data[input_data.diagnostic.name].ordered_channel_list
    #channel_names.append(channel_names[0])
    #pl.xticks(angle_array,channel_names, rotation=90)
    plot_list_4 = []
    for sv_i in range(n_SV):
        col = plot_list_1[sv_i].get_color()
        tmp_topo = join_ends(input_data.topos[sv_i])
        pos,neg =  posNegFill(angle_array,np.zeros(len(angle_array)),tmp_topo)
        ### BUG: it looks like ax4.fill doesn't work in a couple of cases, leaving sub_plot_4_list[i] as int, which raises a set_visible() bug in button_action - also has problems with draw(). other subplots all worked fine before I started with subplot 4
        sub_plot_4_list = list(range(len(pos)+len(neg)+2))
        for j in range(len(pos)):
            sub_plot_4_list[j], = ax4.fill(pos[j][0],pos[j][1],col,visible= button_setting_list[sv_i],alpha=0.5)
        for j in range(len(neg)):
            sub_plot_4_list[j+len(pos)], = ax4.fill(neg[j][0],neg[j][1],col,visible= button_setting_list[sv_i],alpha=0.5)
                
        sub_plot_4_list[len(neg)+len(pos)+0], = ax4.plot(angle_array,tmp_topo,'%so' %(col),visible= button_setting_list[sv_i],markersize=3,label=['', 'Topo {svi}'.format(svi=sv_i)][button_setting_list[sv_i]])
        # show repeated val
        sub_plot_4_list[len(neg)+len(pos)+1], = ax4.plot([angle_array[-1]],[tmp_topo[-1]],'kx', visible= button_setting_list[sv_i],markersize=6)
        debug_(pyfusion.DEBUG, 2, key='svdplot')
        ax4.legend(frameon=0, fontsize='small')
        plot_list_4.append(sub_plot_4_list)

    def test_action(label):
        print(label)

    def button_action(label):
        print('action')
	# this is not very clear: but basically, the label is the str() of the element of plot_list_x we want to make / unmake visible
        visible_status = plot_list_1[int(label)].get_visible()
        plot_list_1[int(label)].set_visible(not visible_status)
        plot_list_2[int(label)].set_visible(not visible_status)
        plot_list_3[int(label)].set_visible(not visible_status)
        for i in range(len(plot_list_4[int(label)])):
            plot_list_4[int(label)][i].set_visible(not visible_status)
        # if visible_status == False, then we are adding visiblity => add to energy, vice-verca
        if visible_status:
            energy.sub(int(label))
        else:
            energy.add(int(label))
        energy_label._text='E = %.2f %%' %(100.*energy.value)
        pl.draw()

    # action when button is clicked
    check.on_clicked(button_action)
    #check.on_clicked(test_action)

    # show plot
    pl.show(block=hold)
    ax_list=[ax1, ax2, ax3, ax4]
    return(ax_list)
    


@register("SVDData")
def plot_fs_groups(input_data, fs_grouper):
    """
    Show what would be generated by the supplied fs_grouper function.

    An example of a grouper funcion is
    pyfusion.data.filters.fs_group_threshold

    The grouper function takes an SVDData instance as its first input and
    returns an iterable where each element is a set of singular value
    indices which define a flucstruc.
    """
    # get the groupings from thr fs_grouper
    fs_indices = fs_grouper(input_data)

    # now get a plot of groupings vs. coherence threshold.

    self_cps = input_data.self_cps()
    # first, create a cache for each cross-correlation
    cp_cache = {}
    index_list = range(len(input_data.svs))
    for i in index_list:
        # yes this doubles up on calculations... optimise later.
        cp_cache[i] = [np.mean(abs(cps(input_data.chronos[i], input_data.chronos[sv])))**2/(self_cps[i]*self_cps[sv]) for sv in index_list]
    grouper_list = []
    threshold_vals = xrange(0,100)
    fs_groups = []
    xy_vals = []
    for t in threshold_vals:
        threshold = 0.01*t
        output_ids_list = []
        output_fs_list = []
        remaining_ids = range(len(input_data.svs))
        while len(remaining_ids) > 1:
            rsv0 = remaining_ids[0]
            tmp_cp = [cp_cache[rsv0][i] for i in remaining_ids]
            filtered_elements = [i for [i,val] in enumerate(tmp_cp) if val > threshold]
            output_fs_list.append(sum([input_data.p[remaining_ids[i]] for i in filtered_elements]))
            output_ids_list.append([remaining_ids[i] for i in filtered_elements])
            del_list = [remaining_ids[i] for i in filtered_elements]
            for i in del_list: 
                remaining_ids.remove(i)
        if len(remaining_ids) == 1:
            output_fs_list.append(input_data.p[remaining_ids[0]])
            output_ids_list.append(remaining_ids)
        fs_groups.append(output_fs_list)
        xy_vals.extend([[threshold, i] for i in output_fs_list])
        
        for i,j in enumerate(output_fs_list):
            if output_ids_list[i] in fs_indices:
                grouper_list.append([threshold, j])


    pl.plot([i[0] for i in xy_vals], [i[1] for i in xy_vals], 'ko')
    pl.plot([i[0] for i in grouper_list], [i[1] for i in grouper_list], 'ro')
    pl.xlim(0,1)
    pl.ylim(0,1)
    pl.grid(True)
    pl.show()
