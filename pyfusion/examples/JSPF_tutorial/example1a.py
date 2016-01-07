#   After running example1.py, you can just do  
# run -i example1a.py
#   (the -i preserves the variables )
# If you omit the -i, you will see this error "name 'myDA' is not defined"
# (or you can paste in into ipython (after running example1.py))

# first, define a convenient shortcut function to plot in a new window
# [you could just use plt.plot() instead]
import matplotlib.pyplot as plt
def pl(array, comment=None,**kwargs):
    plt.figure(num=comment)  # coment written to window title
    plt.plot(array, **kwargs)

# get the variables into local scope - so they can be accessed directly
myDA.extract(locals())        

# the ne profiles are in an Nx15 array, where N is the numer of channels
pl(ne_profile[40,:],'one profile')

# plot a sequence of profiles, showing every fifth 
for prof in ne_profile[10:20:5]:
    pl(prof)

# plot all profiles by using the transpose operator to get profiles
pl(ne_profile.T, 'all profiles',color='b',linewidth=.01)

# without the transpose, you will get the time variation for the data
pl(ne_profile, 'time variation, all channels',color='b',linewidth=.3)

# see all profiles as a false colour image
# time and shot number run vertically, each band is a shot
plt.figure(num = 'image of all data')
plt.imshow(ne_profile,origin='lower',aspect='auto')
# the negative data artifacts near (12,200) are due to fringe skips

plt.show(0)    # needed to make figures appear if "run" instead of pasted.

