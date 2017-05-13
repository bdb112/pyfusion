import pylab as pl
import time as tm

def save_matching(str, prefix = None):
    """ save all pylab windows with str in the title,
    Note that figure(num='myname') is a legal way to name a fig"""
    if prefix is None:
        prefix = tm.strftime('%Y%m%d%H%M')
        print('defaulting to prefix={p}'.format(p=prefix))

    labs = pl.get_figlabels()
    for lab in labs:
        if str in lab:
            pl.figure(lab)
            pl.savefig((prefix+lab).replace(' ','_'))

smw = save_matching  # short cut

def raise_matching(str):
    """ raise all pylab windows with str in the title,
    whether they flash or raise depends on window manager and settings
    Note that figure(num='myname') is a legal way to name a fig"""
    labs = pl.get_figlabels()
    for lab in labs:
        if str in lab:
            pl.figure(lab)
            mgr = pl.get_current_fig_manager()
            mgr.window.tkraise()

rmw = raise_matching  # short cut

def close_matching(str):
    """ close all pylab windows with str in the title,
    see also raise _matching"""
    (labs_nums) = zip(pl.get_figlabels(),pl.get_fignums())
    closed = 0
    for (lab,num) in labs_nums:
        if (str == '' and str == lab) or (str != '' and str in lab):
            # x=input('pause')
            pl.close(num)
            closed += 1
    if closed == 0: print('No figures matching "{s}" found'
                          .format(s=str))

cm = close_matching

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
