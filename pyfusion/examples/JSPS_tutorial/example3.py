# flag all profiles that can't be fitted well by a poly
inds = None
deg = 5
x = np.arange(len(ne_profile[0]))
myDA.extract(locals(),inds=inds)  # make sure they are all numpy variables
err = 0 * t_mid
for (i, nep) in enumerate(ne_profile):
    p = np.polyfit(x, nep,deg, w=nep)
    # the error of the polynomial fit
    err[i] = 1/sqrt(len(nep)) * np.linalg.norm(nep*(nep - np.polyval(p, x)))
    small=0.2  # 0.5 for LHD, 0.2 for H-1
    # discard all profiles with too many small data points
    if len(np.where(nep>small)[0]) < deg:
        err[i]=999
    # and discard profiles that are very small
    if np.average(nep)<small/3:
        err[i]=998

# normalise to the average value
avg = 0 * t_mid
for (i, nep) in enumerate(ne_profile):
    avg[i] = np.average(nep)
    ne_profile[i] = nep/avg[i]

# Plot normalised profiles that are reasonably smooth and not discared above
plt.figure()
for (e,pr) in zip(err,ne_profile):
    if e < small/2:
        plt.plot(pr,color='g',linewidth=.04)
    plt.ylim(0,2)    
plt.show(0)
# the darker areas show recurring profiles.
# We need more information (e.g. power, transform, B_0) to investigate the
# reason for the different profiles.
