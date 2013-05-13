import pylab as pl

def raise_matching(str):
    """ raise all pylab windows with str in the title,
    whether they flash or raise depends on window manager and settings
    Note than figure(num='myname') is a legal way to name a fig"""
    labs = pl.get_figlabels()
    for lab in labs:
        if str in lab:
            pl.figure(lab)
            mgr = pl.get_current_fig_manager()
            mgr.window.tkraise()

rwm = raise_matching
def raise_matching(str):
    """ raise all pylab windows with str in the title,
    whether they flash or raise depends on window manager and settings
    Note than figure(num='myname') is a legal way to name a fig"""
    labs = pl.get_figlabels()
    for lab in labs:
        if str in lab:
            pl.figure(lab)
            mgr = pl.get_current_fig_manager()
            mgr.window.tkraise()

rwm = raise_matching  # short cut

def close_matching(str):
    """ close all pylab windows with str in the title,
    see also raise _matching"""
    (labs_nums) = zip(pl.get_figlabels(),pl.get_fignums())
    closed = 0
    for (lab,num) in labs_nums:
        if str in lab:
            pl.close(num)
            closed += 1
    if closed == 0: print('No figures matching {s} found'
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
