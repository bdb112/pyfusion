import matplotlib
from matplotlib import pyplot as plt
def nicer_log_axis(ax=None, axis_name='x'):
    """ to remove 0.1 etc, the scale must be expanded enough so that 0.6 doesn't round to 1
    (can run again after expanding)
    """ 
    if ax is None:
        ax = plt.gca()
    if axis_name.lower() == 'x':
        ax.set_xscale('log')
        thisax = ax.get_xaxis()
    else:
        ax.set_yscale('log')
        thisax = ax.get_yaxis()

    thisax.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
    tlabs = thisax.get_ticklabels(minor=True)
    # this MUST be before the tlb loop
    thisax.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for tlb in tlabs:
        if len([ch for ch in '0346789' if tlb.get_text().startswith(ch)])>0:
            tlb.set_visible(0)
    # could use tlb.get_unitless_position() to do different things for values < 1 (all 0 on log)
    #[tlb.set_visible(0)  for tlb in tlabs
    # if len([ch for ch in '346789' if tlb.get_text().startswith(ch)])>0]
    #[tlb.set_visible(0) for tlb in tlabs if tlb.get_text().startswith('7') or tlb.get_text().startswith('8') or tlb.get_text().startswith('9')]
    plt.show

#    [tlb.set_visible(0)  for tlb in tlabs if len([tlb.get_text().startswith(ch) for ch in '346789'])]
