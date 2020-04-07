#see list of configs on http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/
"""
# -*- coding: utf-8 -*- ''' Created on Tue Nov 3 12:38:54 2015 @author: micg ''' import matplotlib.pyplot as plt import requests url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0390_+0390/01/020s/' ''' web service calls... ''' fs0 = requests.get(url + 'fluxsurfaces.json?phi=0').json() fs36 = requests.get(url + 'fluxsurfaces.json?phi=36').json() iota = requests.get(url + 'iota.json').json() pressure = requests.get(url + 'pressure.json').json() ''' plotting the results: ''' plt.figure(1) plt.subplot(221) plt.title('flux surfaces at phi = 0Â°') plt.xlabel('R') plt.ylabel('Z') plt.axis([4.5, 6.5, -1.0, 1.0]) plt.plot(fs0['surfaces'][0]['x1'], fs0['surfaces'][0]['x3'], 'r.') for i in range(len(fs0['surfaces'])): plt.plot(fs0['surfaces'][i]['x1'], fs0['surfaces'][i]['x3'], 'r') plt.subplot(222) plt.title('flux surfaces at phi = 36Â°') plt.xlabel('R') plt.ylabel('Z') plt.axis([4.5, 6.5, -1.0, 1.0]) plt.plot(fs36['surfaces'][0]['x1'], fs36['surfaces'][0]['x3'], 'r.') for i in range(len(fs36['surfaces'])): plt.plot(fs36['surfaces'][i]['x1'], fs36['surfaces'][i]['x3'], 'r') plt.subplot(223) plt.title('iota profile') plt.xlabel('radial coordinate s') plt.ylabel('iota') plt.plot(iota['iotaProfile']) plt.subplot(224) plt.title('pressure profile') plt.xlabel('radial coordinate s') plt.ylabel('pressure in Pa') plt.plot(pressure['pressureProfile']) plt.show()

vmec_soap_sample.py
# -*- coding: utf-8 -*- ''' Created on Tue Nov 3 12:57:52 2015 @author: micg ''' import matplotlib.pyplot as plt from osa import Client vmec = Client("http://esb:8280/services/vmec_v5?wsdl") id = 'w7x/1000_1000_1000_1000_+0390_+0390/01/020s' s = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] numPoints = 100 ''' web service calls... ''' fs0 = vmec.service.getFluxSurfaces(id, 0.0, s, numPoints) fs36 = vmec.service.getFluxSurfaces(id, 3.1415/5, s, numPoints) iota = vmec.service.getIotaProfile(id) pressure = vmec.service.getPressureProfile(id) ''' plotting the results: ''' plt.figure(1) plt.subplot(221) plt.title('flux surfaces at phi = 0Â°') plt.xlabel('R') plt.ylabel('Z') plt.axis([4.5, 6.5, -1.0, 1.0]) plt.plot(fs0[0].x1, fs0[0].x3, 'r.') for i in range(len(fs0)): plt.plot(fs0[i].x1, fs0[i].x3, 'r') plt.subplot(222) plt.title('flux surfaces at phi = 36Â°') plt.xlabel('R') plt.ylabel('Z') plt.axis([4.5, 6.5, -1.0, 1.0]) plt.plot(fs36[0].x1, fs36[0].x3, 'r.') for i in range(len(fs36)): plt.plot(fs36[i].x1, fs36[i].x3, 'r') plt.subplot(223) plt.title('iota profile') plt.xlabel('radial coordinate s') plt.ylabel('iota') plt.plot(iota) plt.subplot(224) plt.title('pressure profile') plt.xlabel('radial coordinate s') plt.ylabel('pressure in Pa') plt.plot(pressure) plt.show() 
"""
import matplotlib.pyplot as plt
import requests
import numpy as np
url = 'http://svvmec1.ipp-hgw.mpg.de:8080/vmecrest/v1/geiger/w7x/1000_1000_1000_1000_+0390_+0390/01/020s/'
# this is an EEM case like 20160309,7--11 - 01/020s is beta linear in s, 0.46% beta
#url = url.replace('020s','00s') # beta=0:plasma volume determined by limiter
url = url.replace('020s','00l4') # beta=0:plasma volume larger than limiter
''' web service calls... '''
fs0 = requests.get(url + 'fluxsurfaces.json?phi=0').json()
fs36 = requests.get(url + 'fluxsurfaces.json?phi=36').json()
iota = requests.get(url + 'iota.json').json()
reffs = requests.get(url + 'reff.json').json() # dict of s and reff
# seg7 LP11 daihong gets 0.49715  I get .4963 with 0014, None with 020s and 00s
reff_LP11 = requests.get(url + 'reff.json?x=1.75650&y=-5.40590&z=-0.22140').json() 
print('reff_LP11 = ', reff_LP11)
print('MPM', requests.get(url + 'reff.json?x=-5.6048&y=-2.12504&z=-0.168').json()) 
# loop takes about 12 seconds
reffarray = [[x, requests.get(url + 'reff.json?z=-0.168&y=-2.125&x='+str(-x)).json()] for x in np.linspace(4.8,5.8,endpoint=0)]

from osa import Client
