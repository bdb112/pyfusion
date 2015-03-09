import pylab as pl
import numpy as np

def chunk_plot(x,y,ax=None,plot_centers = False, chunksize=20):

    # Wrap the array into a 2D array of chunks, truncating the last chunk if 
    # chunksize isn't an even divisor of the total size.
    # (This part won't use _any_ additional memory)
    if ax is None:   ax=pl.gca()
    if chunksize is None: chunksize = 10000
    numchunks = y.size // chunksize 
    ychunks = y[:chunksize*numchunks].reshape((-1, chunksize))
    xchunks = x[:chunksize*numchunks].reshape((-1, chunksize))

    # Calculate the max, min, and means of chunksize-element chunks...
    max_env = ychunks.max(axis=1)
    min_env = ychunks.min(axis=1)
    ycenters = ychunks.mean(axis=1)
    xcenters = xchunks.mean(axis=1)

    # Now plot the bounds and the mean...
    ax.fill_between(xcenters, min_env, max_env, color='0.2', 
                    edgecolor='none', alpha=0.5)
    if plot_centers: ax.plot(xcenters, ycenters)
    pl.show()
if __name__ == "__main__":


    num = 1e7
    x = np.linspace(0, 10, num)
    y = np.random.random(num) - 0.5
    y.cumsum(out=y) 
    y += 0.3 * y.max() * np.random.random(num)
    chunk_plot(x,y)
    #fig, ax = plt.subplots()
    
    
