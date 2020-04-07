# From http://webservices.ipp-hgw.mpg.de/docs/howto.html   takes about a minute.
# •Import the osa Client class into your script:
from osa import Client
# •Create a new client object with the URL to the WSDL of the service you want to use:
cl = Client("http://esb.ipp-hgw.mpg.de:8280/services/w7xfp?wsdl")

#•You can now use all web operations of the serives as methods of this client object:
cl.service.getProfilesNumberOfKnots()

#• You can get a listing of all methods of the service by:
print(cl.service)

#• Get information about the data types, that are used in this service by:
print(cl.types)

#• Use the client object to create new objects of such a data type by:
p = cl.types.Points3D()
p.x1 = [5.5, 5.55, 5.6]
p.x2 = [0.0, 0.0, 0.0]
p.x3 = [0.0, 3.14, 6.28]

"""
Full sample script:
from osa import Client
"""
cl = Client("http://esb.ipp-hgw.mpg.de:8280/services/w7xfp?wsdl")

# should be '21'
print(cl.service.getProfilesNumberOfKnots())

pressure = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
jtor = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
coil_currents = [10000,10000,10000,10000,10000,0,0]
plasma_radius = 0.52

rMn = cl.service.getFourierCoefficientsRmn(pressure,jtor,plasma_radius,coil_currents)
zMn = cl.service.getFourierCoefficientsZmn(pressure,jtor,plasma_radius,coil_currents)

iota = cl.service.getIotaProfile(pressure, jtor, plasma_radius, coil_currents)
print(iota)



# Sample 2: Poincare plots from Field Line Tracer web service: 
from osa import Client
import numpy as np
import matplotlib.pyplot as plt

tracer = Client('http://esb.ipp-hgw.mpg.de:8280/services/FieldLineProxy?wsdl')

''' set the start points (R, phi, Z) for the tracing... '''
p = tracer.types.Points3D()
p.x1 = np.linspace(5.64, 6.3, 30)
p.x2 = np.zeros(30)
p.x3 = np.zeros(30)

''' set a coil configuration ... '''
config = tracer.types.MagneticConfig()

''' e.g. using a config ID from CoilsDB: 
    1 : 'w7x standard case', 3 : 'low iota', 4 : 'high iota', 5 : 'low mirror', etc. '''
config.configIds = [1] 

''' you could also create your own coil configurations 
    e.g. use only all type 3 of the non-planar sc coils from w7x: '''
#config.coilsIds = [162, 167, 172, 177, 182, 187, 192, 197, 202, 207]
#config.coilsIdsCurrents = [9993.92, 9993.92, 9993.92, 9993.92, 9993.92, 9993.92, 9993.92, 9993.92, 9993.92, 9993.92]

''' you can use a grid for speeding up your requests. 
    Without a grid all tracing steps will be calculated by using Biot-Savart 
'''
my_grid = tracer.types.CylindricalGrid()
my_grid.RMin = 4.05
my_grid.RMax = 6.75
my_grid.ZMin = -1.35
my_grid.ZMax = 1.35
my_grid.numR = 181
my_grid.numZ = 181
my_grid.numPhi = 481

g = tracer.types.Grid()
g.cylindrical = my_grid
g.fieldSymmetry = 5

config.grid = g


pctask = tracer.types.PoincareInPhiPlane()
pctask.numPoints = 300
pctask.phi0 = [0.0]
                     
task = tracer.types.Task()
task.step = 0.2
task.poincare = pctask

''' you can use a Machine object for a limitation of the tracing region. 
    This sample uses a torus model (id = 164) from ComponentsDB: '''
machine = tracer.types.Machine()
machine.meshedModelsIds = [164] 
machine_grid = tracer.types.CartesianGrid()
machine_grid.XMin = -7
machine_grid.XMax = 7
machine_grid.YMin = -7
machine_grid.YMax = 7
machine_grid.ZMin = -1.5
machine_grid.ZMax = 1.5
machine_grid.numX = 400
machine_grid.numY = 400
machine_grid.numZ = 100

machine.grid = machine_grid
# machine = None

''' make a request to the web service: '''
result = tracer.service.trace(p, config, task, machine, None)

''' plot the results: '''
for i in range(0,len(result.surfs)):
    plt.scatter(result.surfs[i].points.x1, result.surfs[i].points.x3, s=0.1)

plt.show()
