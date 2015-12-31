# this needs to be pasted in after running example1.py, or
# you can just do  
# run -i example1a.py
# the -i preserves the variables 
# if you omit the -i, you will see this error "name 'myDA' is not defined"


# first, define a convenient shortcut function to plot in a new window
# [you could just use plt.plot() instead]
import matplotlib.pyplot as plt
def pl(array, comment=None,**kwargs):
    plt.figure(num=comment)  # coment written to window title
    plt.plot(array, **kwargs)

# get the variables into local scope - so they can be accessed directly
myDA.extract(locals())        

# the ne profiles are in an Nx15 array, where N is the numer of profiles
pl(ne_profile[40,:],'one profile')
# plot all profiles by using a transpose
pl(ne_profile.T, 'all profiles',color='b',linewidth=.01)

# plot a sequence of profiles, showing every fifth 
for prof in ne_profile[5:20:5]:
    pl(prof)

# see all profiles as a false colour image
# time and shot number run vertically, each band is a shot
plt.figure(num = 'image of all data')
plt.imshow(ne_profile,origin='lower',aspect='auto')
# the negative data artifacts near (12,200) are due to fringe skips

plt.show(0)    # needed to make figures appear if "run" instead of pasted.

