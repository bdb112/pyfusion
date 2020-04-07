""" routines for use in 3D graphics
""" 
from numpy import cross, eye, dot
from scipy.linalg import expm, norm

# by B.M. http://stackoverflow.com/questions/6802577/python-rotation-of-3d-vector
def M(axis, theta):
    """ use matrix exponentiation to create a rotational matrix - works for vectors too
    https://en.wikipedia.org/wiki/Matrix_exponential
    """
    return expm(cross(eye(3), axis/norm(axis)*theta))

#from user2525140, http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
import pylab as plt
import numpy as np

def set_axes_equal(ax=None, scale=0.7):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Seems to only work well if the plot window is square - 1:1 aspect ratio
    e.g. figsize=(12,12)

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    if ax is None:
        ax = plt.gca()
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]; x_mean = np.mean(x_limits)
    y_range = y_limits[1] - y_limits[0]; y_mean = np.mean(y_limits)
    z_range = z_limits[1] - z_limits[0]; z_mean = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = scale*0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])
    plt.show(0)

if __name__ == "__main__":
    v, axis, theta = [3,5,0], [4,4,1], 1.2
    M0 = M(axis, theta)

    print(dot(M0, v))
    print 'testing vector rotate: expect [ 2.74911638  4.77180932  1.91629719]'

    fig = plt.figure(num='Demonstrate set_axes_equal')
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    X = np.random.rand(100)*10+5
    Y = np.random.rand(100)*5+2.5
    Z = np.random.rand(100)*50+25

    scat = ax.scatter(X, Y, Z)

    set_axes_equal(ax)
    plt.show(0)

