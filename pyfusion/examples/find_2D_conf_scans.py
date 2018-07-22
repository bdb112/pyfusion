from numpy import unique, max, min, argsort, array, nanmin, nanmax
from pyfusion.data.convenience import between, btw, decimate, his, broaden

""" Find 2D parameter scans - initially for H-1

    So far, a simple scan looking at consecutive runs of length lrun for repeated kH and kV
    We only look at abutting ranges if noverlap==0, one more in between if
      noverlap==2 etc
run pyfusion/examples/sql_plot.py shot,kh_req,kappa_V,ne18_bar  _where="((im2>5000) or (i_main>5000)) and abs(kappa_v) <2 and not(kappa_V is kh_req)"
from pyfusion.data.DA_datamining import Masked_DA, DA
da=DA(dat)
da.extract(locals())
run -i pyfusion/examples/find_2D_conf_scans.py

--> 
 98:   9     5    20     97139- 97288
112:   7     5    20     97851- 97945
 28:   5     5    20     74138- 74160
ind   numkh numkv shots from   to

Then you can do scand[28] to see the shots, ks, and

> plot(ne18_bar[scand[28]['inds']])   # assuming ne18_bar was in the sql query above
Notice that the scan 
#  needs plot_sql run first, so call it a script for now
_PYFUSION_TEST_@@Skip
XX_FUSION_TEST_@@PRE@from pyfusion.data.DA_datamining import Masked_DA, DA; da=DA(dat) ; da.extract(locals())


"""
def span(xs):
    return([float(round(nanmin(xs),3)), float(round(nanmax(xs),3))])


try:
    kh_req  # these lines are just to keep the lint checker quiet
except:
    kh_req = None; kappa_v = None; shot = None; indx = None

noverlap = 20-1
lrun = 20

scans = [[[shot[ind] for ind in inds], unique(kh_req[inds]),unique(kappa_v[inds]), inds]
         for inds in [indx[ind:ind + lrun]
                      for ind in indx[::lrun // (noverlap + 1)]]
         if ((btw(len(unique(kappa_v[inds])), 2, 5)) and
             (btw(len(unique(kh_req[inds])), 2, 10)))]

sorti = argsort([len(scan[2]) for scan in scans]) # sort by number of kv's
sorti = argsort([scan[0][0] for scan in scans]) # sort by number of kv's

header = str('ind   numkh range        numkv range        shots    from   to')
print(header)
for ind, scan in zip(sorti, array(scans)[sorti]):
    print('{ind:3}: {nkh:3} {khr}  {nkv:4}  {kvr} {ns:6}    {mins:6}-{maxs:6}'
          .format(ind=ind, nkh=len(scan[1]), nkv=len(scan[2]), ns=len(scan[0]),
                  mins=min(scan[0]), maxs=max(scan[0]), kvr=span(scan[2]), khr=span(scan[1])))
print(header)
scand = dict([[s, dict(kappa_h=scan[1], kappa_v=scan[2], shot=scan[0], inds=scan[3])] for (s, scan) in enumerate(scans)])
