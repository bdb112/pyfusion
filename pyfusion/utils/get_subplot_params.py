import matplotlib.pyplot as plt
sbp = plt.gcf().subplotpars
subpp = []
for att in 'bottom,top,left,right,wspace,hspace'.split(','):
    if hasattr(sbp, att):
        subpp.append('{att}={val}'
                     .format(att=att, val = round(eval('sbp.'+att),4)))

print('subplots_adjust({parms})'.format(parms=', '.join(subpp)))
