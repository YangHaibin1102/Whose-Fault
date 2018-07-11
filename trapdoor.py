
# coding: utf-8

# The constant extension rate 1e-14 (unit: 1/second) is applied box two sides. Setting extremely large yielding strength, it behaves like a viscoelastic rheology while a relatively small one for yieldstrenghFn produces a plastic behavior.

# In[1]:

import underworld as uw
from underworld import function as fn
import glucifer
uw.matplotlib_inline()

import matplotlib.pyplot as pyplot
pyplot.ion()  # this is need so that we don't hang on show() for pure python runs
import numpy as np
import math
import os
import mpi4py
comm = mpi4py.MPI.COMM_WORLD

#from unsupported.lithopress import lithoPressure
#from unsupported.LMR import *
import unsupported.scaling as sca
from unsupported.scaling import nonDimensionalize as nd


LoadFromFile = False
Elasticity = False

outputPath = os.path.join(os.path.abspath("."),"trapdoor_Pr2e6Vis24/")
inputPath = os.path.join(os.path.abspath("."),"trapdoor_Pr2e6Vis24/")
if uw.rank() == 0:
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
uw.barrier()

if(LoadFromFile == True):
    step = 1340
    fo = open(inputPath+"time"+str(step).zfill(4),"r")
    t = fo.read()
    time = float(t)
else:
    step = 0
    time = 0.  # Initial time

nsteps = 3000

u = sca.UnitRegistry

# Define scale criteria
tempMin = 273.*u.degK 
tempMax = (1500.+ 273.)*u.degK
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2
velocity = 1.*u.centimeter/u.year

KL = 1e3*u.meter
Kt = KL/velocity
KT = tempMax 
KM = bodyforce * KL**2 * Kt**2
K  = 1.*u.mole
lengthScale = 1e3

sca.scaling["[length]"] = KL
sca.scaling["[time]"] = Kt
sca.scaling["[mass]"]= KM
sca.scaling["[temperature]"] = KT
sca.scaling["[substance]"] = K

gravity = nd(9.81 * u.meter / u.second**2)
R = nd(8.3144621 * u.joule / u.mole / u.degK)

elementType = "Q1/dQ0"  
    
resX = 48
resY = 96
materialA = 0
materialB = 1
mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = ( resX, resY), 
                                 periodic    = [False, False]  ) 

#if(LoadFromFile == False):      
minX  = nd(-0.2 * u.kilometer)
maxX  = nd(0.2 * u.kilometer)
minY  = 0.
maxY  = nd(1 * u.kilometer)
dx = 0.04
'''
if(LoadFromFile == True):  
    mesh.load(outputPath+"mesh"+str(step).zfill(4))
    minX = min(mesh.data[:,0])
    maxX = max(mesh.data[:,0])
    minY = min(mesh.data[:,1])
    maxY = max(mesh.data[:,1])
'''

mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = ( resX, resY), 
                                 minCoord    = ( minX, minY), 
                                 maxCoord    = ( maxX, maxY),
                                 periodic    = [False,False]  ) 



velocityField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=mesh.dim )
pressureField    = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
stressField = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=2 )
#velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.
stressField.data[:] = [0.,0.]

#  Boundary conditions
 
# Pure shear with moving  walls — all boundaries are zero traction with 

# In[4]:

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
base   = mesh.specialSets["MinJ_VertexSet"]
top    = mesh.specialSets["MaxJ_VertexSet"]
baseFix = mesh.specialSets['Empty']

allWalls = iWalls + jWalls

meshV = nd(-0*u.centimeter/u.year)
Vy = nd(0.*u.centimeter/u.year)
bufferLayer1 = 0.00001
bufferLayer2 = 0.00001
bufferLayer3 = 0.01

for index in mesh.specialSets["MinI_VertexSet"]:
    velocityField.data[index] = [-meshV, Vy]
    #if mesh.data[index][1]>= maxY - bufferLayer :
    #    velocityField.data[index][1] = Vy * (1-(mesh.data[index][1]-maxY+bufferLayer)/bufferLayer)
    #if mesh.data[index][1] <= bufferLayer :
    #    velocityField.data[index][1] = Vy * mesh.data[index][1]/bufferLayer
        
for index in mesh.specialSets["MaxI_VertexSet"]:
    velocityField.data[index] = [ meshV, 0.]
    #if mesh.data[index][1]<=bufferLayer :
    #    velocityField.data[index][1] = -meshV
      
#for index in mesh.specialSets["MaxJ_VertexSet"]:
#    velocityField.data[index][0] = Vy
    #if mesh.data[index][1]<=bufferLayer :
    #    velocityField.data[index][1] = -meshV
for index in mesh.specialSets["MinJ_VertexSet"]:
    if mesh.data[index][0]<=-dx or mesh.data[index][0]>=dx:
        baseFix+=index
        #stressField.data[index] = [0.,nd(270e6*u.pascal)]
        
def gaussian(xx, centre, width):
    return ( np.exp( -(xx - centre)**2 / width )) 
        
velocityBCs = uw.conditions.DirichletCondition( variable        = velocityField, 
                                                indexSetsPerDof = (iWalls+baseFix, baseFix) )
#nbc      = uw.conditions.NeumannCondition( fn_flux=stressField, variable = velocityField,
                                            #nodeIndexSet = base )

    


swarm  = uw.swarm.Swarm( mesh=mesh,particleEscape=True )
pop_control = uw.swarm.PopulationControl(swarm,aggressive=True,particlesPerCell=60)
previousStress  = swarm.add_variable( dataType="double", count=3 )
#surfaceSwarm = uw.swarm.Swarm( mesh=mesh )
deformationSwarm1 = uw.swarm.Swarm ( mesh=mesh,particleEscape=True  )
deformationSwarm2 = uw.swarm.Swarm ( mesh=mesh,particleEscape=True  )
frictionInf  = swarm.add_variable( dataType="double",  count=1 )
cohesion  = swarm.add_variable( dataType="double",  count=1 )

advector        = uw.systems.SwarmAdvector( swarm=swarm,            velocityField=velocityField, order=2 )
adv_deform1        = uw.systems.SwarmAdvector( swarm=deformationSwarm1,            velocityField=velocityField, order=2 )
adv_deform2        = uw.systems.SwarmAdvector( swarm=deformationSwarm2,            velocityField=velocityField, order=2 )
#advector2       = uw.systems.SwarmAdvector( swarm=surfaceSwarm,     velocityField=velocityField, order=2 )



plasticStrain  = swarm.add_variable( dataType="double",  count=1 )
materialVariable = swarm.add_variable( dataType="int", count=1 )

if(LoadFromFile == True):    
    swarm.load(inputPath+"swarm"+str(step).zfill(4))
    #import ipdb; ipdb.set_trace()
    materialVariable.load(inputPath+"materialVariable"+str(step).zfill(4))   
    velocityField.load(inputPath+"velocityField"+str(step).zfill(4))
    plasticStrain.load(inputPath+"plasticStrain"+str(step).zfill(4))    
    deformationSwarm1.load(inputPath+"deformationSwarm1"+str(step).zfill(4))
    deformationSwarm2.load(inputPath+"deformationSwarm2"+str(step).zfill(4))
    if (Elasticity ==True):
        previousStress.load(inputPath+"previousStress"+str(step).zfill(4))
        
if(LoadFromFile == False): 
    swarmLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=80 )
    swarm.populate_using_layout( layout=swarmLayout )
    '''
    surfacePoints = np.zeros((1000,2))
    surfacePoints[:,0] = np.linspace(minX, maxX, 1000)
    surfacePoints[:,1] = nd(0.8e2 * u.kilometer)
    surfaceSwarm.add_particles_with_coordinates( surfacePoints )
    '''
    countx=21
    county=51
    xcoord = np.linspace(minX, maxX, countx)
    ycoord = np.linspace(minY, maxY, county)

    deformPoints = np.zeros( (county*countx,2))

    for j in range(county):
        for i in range(countx):
            
            n = i+j*countx

            deformPoints[n,0] = xcoord[i] 
            deformPoints[n,1] = ycoord[j]
            
    deformationSwarm1.add_particles_with_coordinates( deformPoints )
    
    countx=21
    county=91

    xcoord = np.linspace(minX, maxX, countx)
    ycoord = np.linspace(minY, maxY, county)
    deformPoints = np.zeros( (county*countx,2))

    for j in range(county):
        for i in range(countx):
            
            n = i+j*countx

            deformPoints[n,0] = xcoord[i] 
            deformPoints[n,1] = ycoord[j]
      
    deformationSwarm2.add_particles_with_coordinates( deformPoints )
    plasticStrain.data[:] = 0.
    
    '''
    from random import *
    np.random.seed(0)
    perm=np.random.permutation(len(swarm.particleCoordinates.data))
    sq=sample(perm, 5000) 
    for index,item in enumerate(sq):
        plasticStrain.data[item] = 1
    
    for index, coord1 in enumerate(swarm.particleCoordinates.data):
        y = coord1[1]

        if (y < minY + bufferLayer2 or y > maxY-bufferLayer2):
            plasticStrain.data[index] = 0.
        
    
    plasticStrain.data[:] = 0.1 * np.random.rand(*plasticStrain.data.shape[:])
    plasticStrain.data[:,0] *= gaussian(swarm.particleCoordinates.data[:,0], 0.0, 0.05) 
    #plasticStrain.data[:,0] *= gaussian(swarm.particleCoordinates.data[:,1], 0.2, 0.05)
    
    
    '''
    faultShape1  = np.array([ (0- dx,minY), (0.+ dx,minY),(0.+ dx,maxY-dx),(0.- dx,maxY-dx)])
    fault1 = fn.shape.Polygon( faultShape1 )
    
    for index in range( len(swarm.particleCoordinates.data) ):
        coord = swarm.particleCoordinates.data[index][:]        
        if (fault1.evaluate(tuple(coord))):
            plasticStrain.data[index] = 2
     
    '''
    core_number = 1000
    circle_core = np.zeros((core_number,2))
    circle_core[:,0] = np.random.uniform(minX,maxX,core_number)
    circle_core[:,1] = np.random.uniform(minY,maxY-2*bufferLayer3,core_number)
    
    def circle_core(x,y,x0,y0):
        return (x-x0)*(x-x0)+(y-y0)*(y-y0)
    
    dx = 1.5*min((maxX-minX)/resX,(maxY-minY)/resY)
    i = 0
    
    while i < core_number:
      
        for index, coord1 in enumerate(swarm.particleCoordinates.data):
            y = coord1[1]
            x = coord1[0]
            if (circle(x,y,circle_core[i,0],circle_core[i,1])< dx * dx):
                plasticStrain.data[index] = 2.        
            
        i = i+1
    
    dx = 1.5*min((maxX-minX)/resX,(maxY-minY)/resY)
    def circle(x,y,x0,y0):
        return (x-x0)*(x-x0)+(y-y0)*(y-y0)
    
    for i in range( len(deformationSwarm1.particleCoordinates.data) ):
        coord = swarm.particleCoordinates.data[i][:]
        for index, coord1 in enumerate(swarm.particleCoordinates.data):
            y = coord1[1]
            x = coord1[0]
            if (circle(x,y,coord[0],coord[1])< dx * dx):
                plasticStrain.data[index] = 2. 

    '''    
    coord = fn.input()
    conditions = [((coord[1] > maxY-bufferLayer3), materialB),
      
                  (       True ,           materialA ) ]


            
            
    materialVariable.data[:] = fn.branching.conditional( conditions ).evaluate(swarm)
    previousStress.data[:] = [0., 0., 0.]

    
C0 = nd(2e6* u.pascal)#2e6
F0 = 0.4
C1 = nd(1.0e5* u.pascal)#1e5
F1 = 0.1

strainLow = 0.5
strainUp = 1.5
cohesion.data[:]  = C0
frictionInf.data[:]  = F0 
    # Drucker-Prager yield criterion
for index, coord in enumerate(swarm.particleCoordinates.data):

    if plasticStrain.data[index]>strainLow  and plasticStrain.data[index]<=strainUp:
        cohesion.data[index] = C0+(C1-C0)/(strainUp-strainLow)*(plasticStrain.data[index]-strainLow)
        frictionInf.data[index]  = F0+(F1-F0)/(strainUp-strainLow)*(plasticStrain.data[index]-strainLow)

    if plasticStrain.data[index] >strainUp:
        cohesion.data[index]  = C1
        frictionInf.data[index]  = F1
yieldStressFnA  = cohesion  + frictionInf * pressureField        

yieldMapDic = {materialA:yieldStressFnA,
                   materialB:nd(1e10*u.pascal)}
   
yieldStressFn = fn.branching.map( fn_key=materialVariable, mapping=yieldMapDic )

densityMap = { materialA: nd(   2700. * u.kilogram / u.metre**3),
               materialB: nd(   1. * u.kilogram / u.metre**3)}

densityFn = fn.branching.map( fn_key=materialVariable, mapping=densityMap )

# And the final buoyancy force function. For the benchmark model, no bouyancy is considered.
z_hat = ( 0.0, 1.0 )      
buoyancyFn = -densityFn * z_hat * gravity


strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)+nd(1e-18/u.second)
upcrustViscosity_Factor      = nd(0.00032 / u.megapascal ** 2.3 / u.second)
upcrustViscosity_n           = 2.3
upcrustViscosity_Q           = nd(154.*u.kilojoule / u.mole)
upcrustViscosity_V           = 0.

def materialStress(factor,n,Q,V,strainrate,pressure,temperature): 
    return fn.math.pow(factor,-1./n)*fn.math.pow(strainrate,(1./n))*fn.math.exp((Q+pressure*V)/(n*R*temperature))
temperatureField = nd(300.*u.degK)

upcrustViscosity = materialStress(upcrustViscosity_Factor,upcrustViscosity_n,\
                upcrustViscosity_Q,upcrustViscosity_V,strainRate_2ndInvariantFn,pressureField,temperatureField)\
                /(2.*(strainRate_2ndInvariantFn))

viscosityA = nd(1e24* u.pascal * u.second) 
#viscosityA = upcrustViscosity
mu = nd(1e10 * u.pascal)                  # elastic modulus
min_viscosity = nd(1e17*u.pascal*u.second)   

if (Elasticity == True):
    mappingDictViscosity1   = { materialA: viscosityA,
                                materialB: nd(1e17*u.pascal*u.second)}
    viscosityMapFn1         = fn.branching.map( fn_key=materialVariable, mapping=mappingDictViscosity1 )
    
    mappingDictMu     = { materialA: mu,
                          materialB: mu}
    muFn   = fn.branching.map( fn_key=materialVariable, mapping=mappingDictMu )
    


    alpha   = viscosityMapFn1/ muFn                         # viscoelastic relaxation time
    dt_e    = nd(3000*u.year)                        # elastic time step
    eta_eff = ( viscosityMapFn1 * dt_e ) / (alpha + dt_e)  # effective viscosity

    mappingDictViscosity   = { materialA: eta_eff,
                               materialB: eta_eff}
    viscosityMapFn   = fn.branching.map( fn_key=materialVariable, mapping=mappingDictViscosity )
    
    
    strainRate_effective = strainRateFn + 0.5*previousStress/(mu*dt_e)
    strainRate_effective_2ndInvariant = fn.tensor.second_invariant(strainRate_effective)+nd(1e-18/u.second)
    yieldingViscosityFn =  0.5 * yieldStressFn / strainRate_effective_2ndInvariant

    
    viscosityFn = fn.exception.SafeMaths( fn.misc.max(fn.misc.min(yieldingViscosityFn, 
                                                                  viscosityMapFn), min_viscosity))


    # contribution from elastic rheology
    tauHistoryFn    = viscosityFn / ( mu * dt_e ) * previousStress 
    # stress from all contributions, including elastic,viscous,plastic (if yielded)
    allStressFn     = 2. * viscosityFn * strainRate_effective#
    # contribution from viscous reholgoy
    viscousStressFn = allStressFn - tauHistoryFn
    # contribution from plastic reholgoy
    #plasticStressFn = allStressFn - tauHistoryFn - viscousStressFn

    mappingDictStress        = {  materialA: allStressFn,
                                  materialB: allStressFn}
    stressMapFn = fn.branching.map( fn_key=materialVariable, mapping=mappingDictStress )

    mappingDictStressElastic = {  materialA: tauHistoryFn,
                                  materialB: tauHistoryFn}
    elasticStressMapFn = fn.branching.map( fn_key=materialVariable, mapping=mappingDictStressElastic )

    mappingDictStressViscous = {  materialA: viscousStressFn,
                                  materialB: viscousStressFn}
    viscousStressMapFn = fn.branching.map( fn_key=materialVariable, mapping=mappingDictStressViscous )
    
    
    stokes = uw.systems.Stokes(    velocityField = velocityField, 
                               pressureField = pressureField,
                               voronoi_swarm = swarm, 
                               conditions    = [velocityBCs,],
                               fn_viscosity  = viscosityFn, 
                               fn_bodyforce  = buoyancyFn,
                               _fn_stresshistory = tauHistoryFn)

if (Elasticity == False): 

    mappingDictViscosity   = { materialA: viscosityA,
                               materialB: min_viscosity}
    viscosityMapFn   = fn.branching.map( fn_key=materialVariable, mapping=mappingDictViscosity )
    
    yieldingViscosityFn =  0.5 * yieldStressFn / strainRate_2ndInvariantFn
    #viscosityFn = fn.exception.SafeMaths( fn.misc.max( 
    #                                                              viscosityMapFn,min_viscosity))
    viscosityFn = fn.exception.SafeMaths( fn.misc.max(fn.misc.min(yieldingViscosityFn, 
                                                                 viscosityMapFn), min_viscosity))
    stressMapFn = 2.0 * viscosityFn*strainRateFn
    
    stokes = uw.systems.Stokes(velocityField = velocityField, 
                               pressureField = pressureField,
                               voronoi_swarm = swarm, 
                               conditions    = [velocityBCs,],
                               fn_viscosity  = viscosityFn, 
                               fn_bodyforce  = buoyancyFn)   
                                         
    



solver = uw.systems.Solver( stokes )


if(uw.nProcs()==1):
    solver.set_inner_method("lu")
else:
    solver.set_inner_method('mumps') 
solver.set_penalty(1.0e-4) 


#solver.options.scr.ksp_rtol = 1.0e-3

# test it out



# In[17]:

surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)

(area,) = surfaceArea.evaluate()
(p0,) = surfacePressureIntegral.evaluate() 

pressureField.data[:] -= p0 / area

mesh_vels = meshV*np.copy(mesh.data[:,0])/maxX

def second_invariant(intensor):
    return fn.math.sqrt(
                0.5*(intensor[0]*intensor[0]+ 
                     intensor[1]*intensor[1]+
                     intensor[2]*intensor[2])
    )                 
time_factor = nd(1*u.year)
def update():
    # get timestep and advect particles
    dt = advector.get_max_dt()
    
    if(Elasticity ==True):
        if dt > ( dt_e / 3. ):
            dt = dt_e / 3. 
            print dt_e/time_factor
            print dt/time_factor  
    print dt/time_factor       
    advector.integrate(dt)
    adv_deform1.integrate(dt)
    adv_deform2.integrate(dt)
    '''
    with mesh.deform_mesh( isRegular=True ):
        mesh.data[:,0] += mesh_vels[:]*dt

    newtime = time + dt
    # recalc mesh exten
    newminX = minX - meshV * newtime
    newmaxX = maxX + meshV * newtime
    '''

    # particle population control
    pop_control.repopulate()
    
    # smoothed stress history for use in (t + 1) timestep 
    if (Elasticity == True):
        phi = dt / dt_e;
        stressDropUnit = [nd(1e6*u.pascal),nd(1e6*u.pascal),nd(1e6*u.pascal)]
        stressMapFn_data = stressMapFn.evaluate(swarm)
        ## reduce stress by 1 MPa once it reaches yielding point
        #swarmYield = viscosityMapFn.evaluate(swarm) > viscosityFn.evaluate(swarm)
        #stressMapFn_data2 = np.where(swarmYield,stressMapFn_data-stressDropUnit, stressMapFn_data )
        #previousStress.data[:] = ( phi*stressMapFn_data2[:] + ( 1.-phi )*previousStress.data[:] )    
        previousStress.data[:] = ( phi*stressMapFn_data[:] + ( 1.-phi )*previousStress.data[:] )
    

    # update plastic strain
    swarmYield = viscosityMapFn.evaluate(swarm) > viscosityFn.evaluate(swarm)
 
    swarmStrainRateInv = strainRate_2ndInvariantFn.evaluate(swarm)
    
    plasticStrainIncrement = dt * np.where(swarmYield, swarmStrainRateInv , 0.0 )
    plasticStrain.data[:] += plasticStrainIncrement
    
    '''
    for index, coord in enumerate(swarm.particleCoordinates.data):
        x = coord[0]
      
        if (x < minX+0.05 or x > maxX-0.05):
            plasticStrain.data[index] = 0.
    '''        
    #newmeshV = (math.sin(newtime/nd(10000*u.year)*3.1415926))*nd(1e-14/u.second)*newmaxX
    #newmeshV = nd(0*1e-14/u.second)*newmaxX
    #if step < 250:
    #   newmeshV = nd(5e-14/u.second)*newmaxX
            
    return time+dt, step+1





while step<nsteps:
    # Obtain V,P and remove null-space / drift in pressure

    solver.solve( nonLinearIterate=True,  nonLinearTolerance=1e-2, nonLinearMaxIterations=15)
    
    (area,) = surfaceArea.evaluate()
    (p0,) = surfacePressureIntegral.evaluate() 
    pressureField.data[:] -= p0 / area
    
    
    if (step%10==0):  
        
        if (Elasticity == True):
            
            meshElasticStress = uw.mesh.MeshVariable( mesh, 1 )
            meshViscousStress = uw.mesh.MeshVariable( mesh, 1 )


            elasticStress = second_invariant(elasticStressMapFn)
            projectorStress = uw.utils.MeshVariable_Projection(meshElasticStress,elasticStress,type=0 )
            projectorStress.solve() 

            viscousStress = second_invariant(viscousStressMapFn)
            projectorStress = uw.utils.MeshVariable_Projection(meshViscousStress,viscousStress,type=0 )
            projectorStress.solve()


            meshElasticStress.save(outputPath+"meshElasticStress"+str(step).zfill(4))
            meshViscousStress.save(outputPath+"meshViscousStress"+str(step).zfill(4))    
            previousStress.save(outputPath+"previousStress"+str(step).zfill(4))
            
        meshViscosity = uw.mesh.MeshVariable( mesh, 1 )
        projectorViscosity = uw.utils.MeshVariable_Projection( meshViscosity,viscosityFn, type=0 )
        projectorViscosity.solve() 
        
        meshAllStress = uw.mesh.MeshVariable( mesh, 1 )
        allStress = fn.tensor.second_invariant(stressMapFn)
        projectorStress = uw.utils.MeshVariable_Projection( meshAllStress, allStress, type=0 )
        projectorStress.solve() 
        


        
        mesh.save(outputPath+"mesh"+str(step).zfill(4))
        meshViscosity.save(outputPath+"meshViscosity"+str(step).zfill(4))
        meshAllStress.save(outputPath+"meshAllStress"+str(step).zfill(4)) 
        swarm.save(outputPath+"swarm"+str(step).zfill(4))
        materialVariable.save(outputPath+"materialVariable"+str(step).zfill(4))
        velocityField.save(outputPath+"velocityField"+str(step).zfill(4))

        pressureField.save(outputPath+"pressureField"+str(step).zfill(4))
        deformationSwarm1.save(outputPath+"deformationSwarm1"+str(step).zfill(4))
        deformationSwarm2.save(outputPath+"deformationSwarm2"+str(step).zfill(4))
        plasticStrain.save(outputPath+"plasticStrain"+str(step).zfill(4))        
        
        fo = open(outputPath+"time"+str(step).zfill(4),"w")
        print >> fo, time
        fo.close()
        
        dicMesh = { 'elements' : mesh.elementRes, 
                    'minCoord' : mesh.minCoord,
                    'maxCoord' : mesh.maxCoord}

        fo = open(outputPath+"dicMesh"+str(step).zfill(4),'w')
        fo.write(str(dicMesh))
        fo.close()       
        
    if uw.rank()==0:   
        print('step = {0:6d}; time = {1:.3e};'.format(step,time/time_factor))
    uw.barrier()
    
    # finished timestep, update all
    time, step = update()


