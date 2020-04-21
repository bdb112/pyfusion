import pyfusion
import numpy as np

"""
From Philipp Drew, April 2020

x0=-5930.11862;
y0=-2254.21244;
z0=-168;

x1=-5604.82757;
y1=-2125.03934;
z1=-168;

%open position signal
[r] = mdsvalue('QRN.HARDWARE:ACQ132_170:INPUT_15');

%open time signal for position measurement
[tr] = mdsvalue('DIM_OF(QRN.HARDWARE:ACQ132_170:INPUT_15)');

veclen=sqrt((x1-x0)^2+(y1-y0)^2+(z1-z0)^2);
rpos(:,j)=-r*0.05;
xpos(:,j)=x0-rpos(:,j)*(x1-x0)/veclen;
ypos(:,j)=y0-rpos(:,j)*(y1-y0)/veclen;
zpos(:,j)=z0-rpos(:,j)*(z1-z0)/veclen;
# See below - had to divide first term by 1e3
"""

def get_MPM_data(shot_number=[20160309,32]):
    """ retrieve MPM x,y,z from hardware channel 15. Note that the sample rate is 5MS/s
    so we should use downsampling of some sort
    """
    dev_name = 'W7M'
    dev = pyfusion.getDevice(dev_name)
    data = dev.acq.getdata(shot_number, 'W7M_MPM_R')

    MPM_r = data.signal
    
    x0, y0, z0 = -5930.11862, -2254.21244, -168
    x1, y1, z1 = -5604.82757, -2125.03934, -168

    veclen = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2);
    MPM_r = -1 * MPM_r
    xpos = x0/1e3 - MPM_r * (x1-x0)/veclen
    ypos = y0/1e3 - MPM_r * (y1-y0)/veclen
    zpos = z0/1e3 - MPM_r * (z1-z0)/veclen
    return(np.array([xpos, ypos, zpos])/1e0)
