""" Plot data from a dd or DA.da dictionary - useful as a rough idea of the shot.
This version allows arbitrary python expressions of one variable.
"""

import pylab as pl
import numpy as np
from numpy import *    # this allows expressions like sin in the diags 


# first, arrange a debug one way or another
try: 
    from pyfusion.debug_ import debug_
except:
    def debug_(debug, msg='', *args, **kwargs):
        print('attempt to debug ' + msg + 
              " need boyd's debug_.py to debug properly")


def plot_shots(da, shots=None, nx=6, ny=4, diags=None, fontsize=None, marker=None, extra_diags=None,  save='', quiet=True, **kwargs):
    """ Call plot_shot to produce a matrix of plots, from data in a dictionary of
    arrays, optionally saving them in png files.  The arrays in the dictionary 
    should be the same size.  Originally intended for the DA.da dictionary,
    but works for any such dictionary.
    """ 
    # four shots/sec saving to 18x12 png 
    plot_shots.__doc__ += plot_shot.__doc__ 

    if fontsize != None:     
        pl.rcParams['legend.fontsize'] = fontsize

    if shots is None: shots = np.unique(da['shot'])

    #for (s,sh) in enumerate(shots[0:nx*ny]): 
    for (s,sh) in enumerate(shots): 
        if np.mod(s+1, ny*nx) == 0: 
            pl.gcf().canvas.set_window_title('Shots {f} to {t}'
                                             .format(t=sh, f=sh-nx*ny))
            if save == '':  pl.figure()
            else: 
                if s>0: 
                    f=pl.gcf()
                    f.set_size_inches(18,12)
                    f.savefig('{p}{f}_{t}'
                              .format(p=save,t=sh, f=sh-nx*ny))
                    pl.clf()

        pl.subplot(nx,ny,np.mod(s,nx*ny)+1)
        plot_shot(da, sh, marker=marker, extra_diags=extra_diags, fontsize=fontsize, diags=diags, quiet=quiet, **kwargs)

        if nx*ny>4:
            pl.subplots_adjust(.02,.03,.97,.97,.24,.13)

def safe_get(da,key,inds=None):
    try:
        val = da[key][inds]
    except:
        val = len(inds) * [None]
    return(val)

def eval_diag(da, inds, diag, debug=0):
    # put keys into order of decreasing length to avoid false hits (e.g. p instead of w_p)
    longest_first_order = np.argsort([-len(k) for k in da.keys()])
    keys_longest_first = [da.keys()[k] for k in longest_first_order]
    separators = [chr(i) for i in range(40,128)]
    # remove all but alphanum and _  - for now, do it the easy way
    separators='{}[]!@#$%^&*()_+-=<>:?/\|~`'
    diag_stripped = diag
    for sep in separators: 
        diag_stripped = diag_stripped.replace(sep,' ')
 
    # for each key found, create a local of that name populated by [inds]
    for k in keys_longest_first:
        if debug>1: print(k),
        if k in diag_stripped:
            exec("{k} = da['{k}'][inds]".format(k=k))
            # remove it form the stripped expression
            diag_stripped = diag_stripped.replace('k','')
            if debug>0: print(k, k in locals())

        if debug>1: 
            print('input {d}, -> actual {a}, key {k}, list{l}'
                  .format(d=diag, a=actual_diag, k=k, l=keys_longest_first))

    try:
        exec('dat='+diag)
        return(None, dat)  # None is good
        
    except Exception as reason:  # NameError 
        str_reason = str('{d} could not be evaluated{r}, {a}'
                         .format(r=reason, d=diag, a=reason.args))
        if debug>0: raise ValueError(str_reason)
        else: return(reason, (str_reason, diag_stripped))

def plot_shot(da, sh=None, ax=None, diags = None, marker=None, extra_diags=None, debug=0, quiet=True, fontsize=None, hold=1, **kwargs):
    """ more flexible - no need to check for errors
    Also initial version tailored to LHD, but a fudge used to detect HJ
    extra_diags are simply added to the default list.  This way a default list
    can be used simultaneously with a one or more extra diags.
    """

    if fontsize is not None:     
        pl.rcParams['legend.fontsize']=fontsize

    if diags is None:
        if 'MICRO01' in da.keys():
            diags = 'MICRO01,DIA135'
        else:
            diags = 'i_p,w_p,flat_level,NBI,ech,p_frac'

    if extra_diags is not None:
        diags += "," + extra_diags

    if sh is None: sh = da['shot'][0]

    inds=np.where(sh==da['shot'])[0]
    pl.rcParams['legend.fontsize']='small'
    if ax is None: 
        if hold==0: pl.plot(hold=0)
        ax = pl.gca()
    #(t_mid,w_p,dw_pdt,dw_pdt2,b_0,ech,NBI,p_frac,flat_level)=9*[None]
    t = da['t_mid'][inds]
    b_0 = safe_get(da, 'b_0',inds)

    if (len(np.shape(diags)) == 0): diags = diags.split(',')
    for (i,diag) in enumerate(diags):
        lab = diag   # set default label, marker
        if marker is None: marker = '-' 

        (err, dat) = eval_diag(da=da, inds=inds, diag=diag, debug=debug)
        if quiet and err is not None:
            continue

        if diag in 'p_frac': 
            dat=dat*100
            lab+="*100"
        elif diag in 'ech': 
            dat=dat*10
            lab+="*10"
        elif 'flat_level' in diag: 
            dat = 30+200*dat
            marker='--'

        if diag == 'p_frac': 
            marker = ':'    

        ax.plot(t,dat, marker, label=lab, **kwargs)
        ## no hold keyword in ax.plot #, hold=(hold & (i==0)))
    pl.legend()
    debug_(debug,1,key='plot_shot')

    pl.title("{s} {b:.3f}T".format(s=sh,b=b_0[0]))

def plot_shotold(DA, sh, ax=None):
    pl.rcParams['legend.fontsize']='small'
    if ax is None: ax = pl.gca()
    #(t_mid,w_p,dw_pdt,dw_pdt2,b_0,ech,NBI,p_frac,flat_level)=9*[None]
    print('locals before has {n} keys'.format(n=(locals().keys())))
    (t_mid,w_p,dw_pdt,dw_pdt2,b_0,ech,NBI,p_frac,flat_level)=\
        DA.extract(varnames=
                   't_mid,w_p,dw_pdt,dw_pdt2,b_0,ech,NBI,p_frac,flat_level',
                   inds=np.where(sh==DA.da['shot'])[0])
    print('locals after has {n} keys'.format(n=(locals().keys())))
    err = []
    ax.plot(t_mid,w_p,label='wp')
    #ax.plot(t_mid,dw_pdt2,linewidth=0.1,label='d_wpdt2')
    ax.plot(t_mid,10+300*flat_level,':',label='flat_level')
    try:
        ax.plot(t_mid,100*p_frac,':',label='p_frac')
    except:
        err.append('p_frac')
    try:
        ax.plot(t_mid,NBI,label='NBI')
    except:
        err.append('NBI')
    try:
        ax.plot(t_mid,100*ech,'orange',label='ech')
    except:
        err.append('ech')
    pl.legend()
    pl.title("{s} {b}T".format(s=sh,b=b_0[0]))
    if len(err)>0: print("Error: {e}".format(e=err))
