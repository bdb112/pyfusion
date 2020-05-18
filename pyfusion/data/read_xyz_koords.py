import numpy as np
from matplotlib import pyplot as plt


def read_xyz_koords(filename="/home/bdb112/pyfusion/working/pyfusion/pyfusion/acquisition/W7X/QXM_xyz_data.txt", debug=False, dtype= [('name', 'S100'), ('x', 'f8'), ('y', 'f8'), ('z', 'f8')]):
    """ Read in as a numpy object
         DD = read_xyz_koords()
         # get QXM41 12..37
         xyz = [np.round((dd['x'], dd['y'], dd['z']),7)
               for dd in DD if 'QXM41' in dd['name'] and int(dd['name'][-3:-1]) in range(12,38)]

    """
    dd = np.loadtxt(filename, dtype=dtype, skiprows=1)
    if debug>0:
        1/0
    return(dd)
    
if __name__ == '__main__':

    DD = read_xyz_koords()
    # get QXM41 12..37
    xyz = [np.round((dd['x'], dd['y'], dd['z']), 7)
           for dd in DD if 'QXM41' in dd['name'] and int(dd['name'][-3:-1]) in range(12, 38)]

    from mpl_toolkits.mplot3d import Axes3D
    fig_3D = plt.figure()
    ax3d = fig_3D.add_subplot(111, projection='3d')
    ax3d.plot(*np.array(xyz).T, marker='.')
    plt.show()
