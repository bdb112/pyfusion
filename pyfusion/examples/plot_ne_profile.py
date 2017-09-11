import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import __version__ as pltver
import MDSplus as MDS

_var_defaults="""
shot = 97446 # in bdb local 97440:rising, 93431:missing chans
clip = (0,4)
may = 0

zrange = None
resx = None
resy = None
##chans = range(1, 11)[::-1]
## chans.extend(range(12, 21))
chans = range(1, 21 + 1) # MDS type (1-based) indices
figsizeins = [12,9]
tpoints=None  # 200 needed for mayavi, 100 for axes3D
show_lines=True  # if True, project lines from 3D plot ontothe wall
t_range=None #  confine analysis: e.g. t_range=[0,0.01]
"""

exec(_var_defaults)
from pyfusion.utils import process_cmd_line_args
exec(process_cmd_line_args())

if tpoints is None:
    tpoints = 200 if may else 100

if resx is None:
    if may:
        resx=1000j; resy=1000j # needed for mayavi (wants x and y res the same
    else:
        resx=500j; resy=50j

tr = MDS.Tree('electr_dens', shot)
zlist = tr.getNode('\electr_dens::top.ne_het.chan_z').data()
digs = tr.getNode('\electr_dens::top.ne_het.chan_list').data()
ne_chans = ['ne_{ch:d}'.format(ch=ch) for ch in chans]

bads=[11]
chans = np.array([ch for ch in chans if ch not in bads])

""" looking at the code in process interfermeter and the table in OperationalDiagnostics/Interferometer
first and second elements in z and chan_list are 
Z   ch  digs    
0.0 1 A14_21:1 TR612_7:1  ne_1     python 0 
25  2 A14_21:2 TR612_7:2  ne_2     python 1  
twelfth is
-25.  12  A14_22:6  TR612_8:6 ne_12 (python subsript 11)
"""
ne = []
zused = []

myNoData = MDS.TreeNODATA if hasattr(MDS, 'TreeNODATA') else MDS.TreeNoDataException
TdiException = Exception #  ()  I think Tom is deprecating TdiException in favour of more specific ones

for ch in chans:
    nd = tr.getNode('\electr_dens::top.ne_het.' + ne_chans[ch - 1])
    try:
        if len(ne) == 0:
            tb = nd.dim_of().data()
            if t_range is not None:
                wtr = np.where((tb >= t_range[0]) & (tb <= t_range[1]))[0]
                tb = tb[wtr]

        ne.append(nd.data().clip(*clip))
        if t_range is not None:
            ne[-1] = ne[-1][wtr]

        zused.append(zlist[ch - 1])
    # MDS.TreeNoDataException before 7.1.13
    
    except (TdiException, myNoData) as reason:
        print('{r} error reading {n}'.format(r=reason, n=nd.getNodeName()))
        pass
zused = np.array(zused)
ne = np.array(ne)

inds = np.argsort(zused)

# dec = 500
dec = int(len(tb)/float(tpoints))  # 200 needed for mayavi
tb_sub = tb[0::dec]
ne_sub = ne[inds][:,0::dec]
ne_sub_flat = ne_sub.flatten()
z = zused[inds]
zt = np.array(zip( np.array(len(z) * [tb_sub.tolist()]).flatten() , np.transpose(len(tb_sub) * [z.tolist()]).flatten()))

from scipy.interpolate import griddata
zgrid, tgrid = np.mgrid[z.min(): z.max(): resy, tb_sub.min(): tb_sub.max() : resx]
gridded_ne = griddata(zt, ne_sub_flat, (tgrid, zgrid), method='linear')

titl = str('Shot {shot}'.format(shot=shot))


if may == 0:
    fig, [axupper, axmiddle, axlower] = plt.subplots(nrows=3, ncols=1)
    axupper.contour(tb_sub, z, ne_sub)
    axupper.scatter(zt[: ,0], zt[:, 1], 30*ne_sub_flat, 30*ne_sub_flat, lw=0 )

    interp = 'nearest' # 
    interp = 'bicubic' # 
    #interp = 'lanczos'
    """ 'none', 'nearest', 'bilinear', 'bicubic',
        'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser',
        'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc',
        'lanczos'
    """

    # use zlist here so that the bounds don't change with the shot
    axmiddle.imshow(ne[inds, :],aspect='auto',interpolation=interp, extent=[tb.min(), tb.max(), zused.min(), zused.max()],origin='lower')
    if zrange is not None:
        plt.ylim(zrange.min(), zrange.max())
    else:
        plt.ylim(zlist.min(), zlist.max())

    imdata = axlower.imshow(gridded_ne,aspect='auto',interpolation=interp, extent=[tb.min(), tb.max(), zused.min(), zused.max()], origin='lower')
    fig.colorbar(imdata, fraction=0.1, pad=0.05)
    fig.subplots_adjust(right=1)
    fig.suptitle(titl)

    plt.show(0)

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter


    fig3D = plt.figure(figsize=figsizeins)
    ax = fig3D.gca(projection='3d')
    ax.view_init(20,210)
    # Plot the surface.
    # cm.plasma only in the newest                        # bwr, #coolwarm,
    # tri_surf does not need re-gridded data, but the triangle shading doesn't work
    # surf = ax.plot_trisurf(zt[:,0], zt[:,1], ne_sub_flat, cmap=cm.rainbow,
    #                       linewidth=0, antialiased=False, shade=True)
    mymap = cm.jet if pltver<'1.0.0' else cm.rainbow if pltver<'1.5.0' else cm.plasma
    surf = ax.plot_surface(tgrid, zgrid, gridded_ne, cmap=mymap,
                           linewidth=0, antialiased=False, shade=False, rstride=1)

    # Customize the z axis.
    ax.set_zlim(0, gridded_ne.max())
    ax.set_ylim(-0.3,0.3)
    xlim = ax.get_xlim()
    xlim[1] *= 1.03
    ax.set_xlim(xlim)  # fix the axies
    if show_lines:
        ax.plot(tb_sub,0.3 + tb_sub*0, ne_sub[10,:], 'b', lw=0.3)
        wmax = np.unravel_index(ne_sub.argmax(), ne_sub.shape)
        # thin lines and small dots to mark actual points
        ax.plot(xlim[1] + 0 * z, z,  ne_sub[:,wmax[1]] , 'b.-', lw=0.3, ms=2)

    #ax.zaxis.set_major_locator(LinearLocator(5))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig3D.colorbar(surf, shrink=0.35, aspect=8, fraction=0.1, pad=0.05)
    fig3D.subplots_adjust(right=1)
    ax.set_title(titl)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('Z (m)')
    ax.set_zlabel('$n_e (10^{18}m^{-3})$')

    fig3D.show(0)

else:
    from mayavi import mlab
    from mayavi.version import version as mayver
    from matplotlib.colors import LightSource

    ## Mayavi need .T to get the column/row order right - otherwise crash or hang
    maymap = 'jet' if mayver < '4.5.0' else 'rainbow'
    surf = mlab.surf(-tgrid.T, zgrid.T, gridded_ne.T, colormap=maymap, warp_scale='auto')
    # Change the visualization parameters.
    surf.actor.property.interpolation = 'phong'
    surf.actor.property.ambient = 0.05
    surf.actor.property.diffuse = 0.1
    mlab.view(azimuth=45,elevation=80)
    #surf.actor.property.interpolation='gouraud'
    #surf.actor.property.specular = 0.1
    #surf.actor.property.specular_power = 5

    if mayver<'4.5.0': 
        print('Close mayavi window to return to prompt as version is {mv}'
              .format(mv=mayver))
        mlab.show()
