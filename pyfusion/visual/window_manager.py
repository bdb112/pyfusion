from __future__ import print_function
import pylab as pl
import time as tm
import numpy as np

def get_fig_nums_labs(match_str=''):
    """ get all figure labels if they exist, or str of the fig num if note
    The match_str is compared with lab just to detect no matches - all labs are returned
    """
    fig_nums = pl.get_fignums()
    fig_labs = pl.get_figlabels()
    nums_labs = []
    for num, lab in zip(fig_nums, fig_labs):
        if lab == '': lab = str(num)
        nums_labs.append(lab)
    # check for matches here, save duplicating code.
    matches = [nl for nl in nums_labs if match_str in nl]
    if len(matches) == 0:
        print('No matches to ' + match_str)
    return(nums_labs)

def save_matching(str, prefix = None):
    """ save all pylab windows with str in the title,
    Note that figure(num='myname') is a legal way to name a fig"""
    if prefix is None:
        prefix = tm.strftime('%Y%m%d%H%M')
        print('defaulting to prefix={p}'.format(p=prefix))

    labs = pl.get_fig_nums_labs(str)
    for lab in labs:
        if str in lab:
            pl.figure(lab)
            pl.savefig((prefix+lab).replace(' ','_'))

smw = save_matching  # short cut

def list_matching(str=''):
    """ list all pylab windows with str in the title,
    Note that figure(num='myname') is a legal way to name a fig"""
    labs = [lab for lab in get_fig_nums_labs(str)  if str in lab]
    labs = np.sort(labs)
    # perhaps nicer to make separate lists of fignums and names
    # or use list(map(plt.figure, plt.get_fignums()))

    for lab in labs:
        if lab.isdigit():  # must use int otherwise a new figure is opened
            fig = pl.figure(int(lab))
        else:
            fig = pl.figure(lab)
        if len([ax for ax in fig.get_children() if  hasattr(ax,'get_ylabel')]) ==0:
            print('{lab} <null>'.format(lab=lab), end=', ')
            continue
        print('Figure {n}: {nm}'.format(n=fig.number, nm=lab), end=': ')
        print([ch.get_text() for ch in fig.get_children()
               if hasattr(ch,'get_text') and ch.get_text() != ''])
        print([ax.get_ylabel() for ax in fig.get_children()
               if hasattr(ax,'get_ylabel') and ax.get_ylabel() != ''])
         
        
lmw = list_matching

def raise_matching(str):
    """ raise all pylab windows with str in the title,
    whether they flash or raise depends on window manager and settings
    Note that figure(num='myname') is a legal way to name a fig"""
    labs = [lab for lab in get_fig_nums_labs(str) if str in lab]
    labs = np.sort(labs)
    for lab in labs:
        pl.figure(lab)
        mgr = pl.get_current_fig_manager()
        mgr.window.tkraise()

def order_matching(str, startx=10, starty=10, incx=300, incy=0, wsize=None):
    """ reorder all pylab windows with str in the title,
    whether they flash or raise depends on window manager and settings
    Note that figure(num='myname') is a legal way to name a fig"""
    labs = [lab for lab in get_fig_nums_labs(str) if str in lab]
    if len(labs) == 0:
        print('No figures matching "{s}" found'.format(s=str))
    labs = np.sort(labs)
    x = startx; y=starty
    mgr = pl.get_current_fig_manager()
    screen_y = mgr.window.winfo_screenheight()
    screen_x = mgr.window.winfo_screenwidth()
     
    for (i, lab) in enumerate(labs):
        pl.figure(lab)
        geom = ''
        mgr = pl.get_current_fig_manager()
        w, h = mgr.canvas.get_width_height()
        if wsize is not None:
            w = wsize[0]
            h = wsize[1]
        geom += '{w}x{h}'.format(w=w, h=h)
        if w + x > screen_x:
            x = startx
            y += h
        geom += '+{x}+{y}'.format(x=x, y=y)
        mgr.window.wm_geometry(geom)
        x += incx
        y += incy
        mgr.window.tkraise()
        
omw = order_matching  # short cut
rmw = raise_matching  # short cut

def close_matching(str):
    """ close all pylab windows with str in the title,
    see also raise _matching"""
    # coded differently as the number must be given to close.
    (labs_nums) = zip(pl.get_figlabels(),pl.get_fignums())
    closed = 0
    for (lab,num) in labs_nums:
        if (str == '' and str == lab) or (str != '' and str in lab):
            # x=input('pause')
            pl.close(num)
            closed += 1
    if closed == 0: print('No figures matching "{s}" found'
                          .format(s=str))

cmw = close_matching

if __name__ == "__main__":
    from time import sleep
    figlist=[] # not normally needed - so we can clean up the test 


    for name in 'eeny meeny miney moe'.split():
        figlist.append(pl.figure(num=name))
        pl.title(name)
    pl.show()

    sleep(2)

    print('now cover them up')
    for i in range(10):
        figlist.append(pl.figure())

    msg="raise eeny and meeny using:\n\n raise_matching('eny') \n\n - wait 3 secs"    
    pl.text(0.5,0.5,msg,ha='center', size='x-large')
    pl.show()

    print(msg)
    sleep(3) 

    raise_matching('eeny')
    pl.show()

    x=raw_input('return to clean up')
    for fig in figlist: 
        pl.close(fig)
