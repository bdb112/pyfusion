import pylab as pl
import numpy as np

# first, arrange a debug one way or another
try: 
    from pyfusion.debug_ import debug_
except:
    def debug_(debug, msg='', *args, **kwargs):
        print('attempt to debug ' + msg + 
              " need boyd's debug_.py to debug properly")


def plot_shots(DA, shots=None, nx=6, ny=4, diags=None, fontsize=None, save=''):
    """ four shots/sec saving to 18x12 png 
    """ 
    if fontsize != None:     
        pl.rcParams['legend.fontsize']=fontsize

    if shots is None: shots = np.unique(DA.da['shot'])

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
        plot_shot(DA, sh, diags=diags)

        if nx*ny>4:
            pl.subplots_adjust(.02,.03,.97,.97,.24,.13)

def safe_get(DA,key,inds=None):
    try:
        val = DA.dd[key][inds]
    except:
        val = len(inds) * [None]
    return(val)

def plot_shot(DA, sh=None, ax=None, diags = None, extra_diags=None, debug=0, fontsize=None, hold=1):
    """ more flexible - no need to check for errors
    Also initial version tailored to LHD
    """
    if fontsize is not None:     
        pl.rcParams['legend.fontsize']=fontsize

    if diags is None:
        if 'MICRO01' in DA.da.keys():
            diags = 'MICRO01,DIA135'
        else:
            diags = 'i_p,w_p,flat_level,NBI,ech,p_frac'

    if extra_diags is not None:
        diags += "," + extra_diags

    if sh is None: sh = DA.da['shot'][0]

    inds=np.where(sh==DA.da['shot'])[0]
    pl.rcParams['legend.fontsize']='small'
    if ax is None: 
        if hold==0: pl.plot(hold=0)
        ax = pl.gca()
    #(t_mid,w_p,dw_pdt,dw_pdt2,b_0,ech,NBI,p_frac,flat_level)=9*[None]
    t = DA.da['t_mid'][inds]
    b_0 = safe_get(DA, 'b_0',inds)

    if (len(np.shape(diags)) == 0): diags = diags.split(',')
    for (i,diag) in enumerate(diags):
        if diag in DA.keys:
            dat = DA.da[diag][inds] ; lab = diag; linestyle='-'
            if diag in 'p_frac': 
                dat=dat*100
                lab+="*100"
            elif diag in 'ech': 
                dat=dat*10
                lab+="*10"
            elif 'flat_level' in diag: 
                dat = 30+200*dat
                linestyle='--'
                
            if diag == 'p_frac': linestyle = ':'    
            ax.plot(t,dat, linestyle=linestyle, label=lab)
            ## no hold keyword in ax.plot #, hold=(hold & (i==0)))
    pl.legend()
    debug_(debug,1,key='plot_shot')

    pl.title("{s} {b}T".format(s=sh,b=b_0[0]))

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
