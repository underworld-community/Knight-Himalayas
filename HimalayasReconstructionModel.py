# %% [markdown]
# ### Himalayas reconstruction model
# Includes:
# - Surface processes (D parameter)
# - Internal heating 

# %%
import underworld as uw
import underworld.function as fn

#### import depends on the UW version
if int(uw.__version__[2:4]) > 12:
    from underworld import UWGeodynamics as GEO
else:
    import UWGeodynamics as GEO
    
    
import os
from datetime import datetime
import pytz
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

import underworld.visualisation as visualisation

# %%
if uw.mpi.rank == 0:
    print(f'UWGeo version: {GEO.__version__}')
    print(f'UW version: {uw.__version__}')


# %%
u = GEO.UnitRegistry

# GEO.rcParams["initial.nonlinear.tolerance"] = 1e-2
# GEO.rcParams["nonlinear.tolerance"]         = 1e-2


GEO.rcParams['initial.nonlinear.max.iterations'] = 100
GEO.rcParams['nonlinear.max.iterations']         = 100

GEO.rcParams["popcontrol.particles.per.cell.2D"] = 20
GEO.rcParams["swarm.particles.per.cell.2D"]      = 20

GEO.rcParams["popcontrol.split.threshold"]       = 0.1

# %%
restart = False

# %%
# ### Values to change

#### 3 options
## fast = 10 cm/yr
## slow = 2 cm/yr
## F2S = 10 to 2 cm/yr, this is the default

model_velProfile = 'F2S' # , 'slow', 'fast'

### number of timesteps from each model
noOfTS = 100.

### Total convegence of all models
totalConvergence = 2000.0 ### in km

#### km
Sticky_air = 40.0
x_box      = 1792.0
y_box      = 224.0 - Sticky_air

### thickness of crust (km)
crustalthickness = 25.0

#### transition from lower to upper crust
crust_transition = 650.0

#### start x coordinate of material close to the far wall
back_stop_x_coord = 1350.0

#### updates incoming material on LHS of box (km)
Update_material_LHS_Length = 200.0

#### stops stain weakening near the LHS of box (km)
x_threshold_Strain_weakening = 200.0

#### diffusion constant for surface, m^2/year for diffusion erosion model
D_surface = (150.0 * u.meter**2 / u.year)

### sedimentation and erosion rates for vel erosion model
# Vs = ((0.03 * u.millimeter / u.year).to(u.kilometer / u.year))
# Ve = ((-0.3 * u.millimeter / u.year).to(u.kilometer / u.year))

### reference density values, individual values for materials can also be changed below
sed_dens    = 2500 * u.kilogram/u.meter**3
crust_dens  = 2700 * u.kilogram/u.meter**3
mantle_dens = 3300 * u.kilogram/u.meter**3

### reference IH values, individual values for materials can also be changed below
sediment_IH = 2.   * u.microwatt / u.meter**3
crust_IH    = 2.   * u.microwatt / u.meter**3
mantle_IH   = 0.02 * u.microwatt / u.meter**3

# %%
# ### Setup of box

u = GEO.UnitRegistry

Depth_of_box = y_box * u.kilometer
model_length = x_box * u.kilometer


if uw.mpi.size == 1:
    ### Local res
    x_res = 2**8
    y_res = 2**5
else:
    #### HPC res
    x_res  = 2**10
    y_res =  2**7

if uw.mpi.rank == 0:
    print(x_res,y_res)


# %%
### Scaling

half_rate = 1. * u.centimeter / u.year
length_scale = 300.0 * u.kilometer
surfaceTemp = 273.15 * u.degK
baseModelTemp = 1573.15 * u.degK
bodyforce = 3300 * u.kilogram / u.metre**3 * 9.81 * u.meter / u.second**2

KL = length_scale
Kt = KL / half_rate
KM = bodyforce * KL**2 * Kt**2
KT = (baseModelTemp - surfaceTemp)

GEO.scaling_coefficients["[length]"] = KL
GEO.scaling_coefficients["[time]"] = Kt
GEO.scaling_coefficients["[mass]"]= KM
GEO.scaling_coefficients["[temperature]"] = KT



### Scaling
if uw.mpi.rank == 0:
    print('Length, km = ', GEO.dimensionalise(1., u.kilometer))
    print('Time, Myr = ',GEO.dimensionalise(1., u.megayear))
    print('Pressure, MPa = ',GEO.dimensionalise(1., u.megapascal))
    print('Temperature, K = ',GEO.dimensionalise(1., u.degK))
    print('Mass, kg = ',GEO.dimensionalise(1.,u.kilogram))

    print('Velocity, cm/yr = ',GEO.dimensionalise(1., u.centimeter / u.year))
    print('Diffusivity, m^2/s = ',GEO.dimensionalise(1.,u.metre**2 / u.second))
    print('Density, kg/m^3 = ',GEO.dimensionalise(1.,u.kilogram / u.metre**3))
    print('Viscosity, Pa S = ',GEO.dimensionalise(1.,u.pascal * u.second))
    print('gravity, m/s^2 = ',GEO.dimensionalise(1.,u.meter / u.second**2))




# %% [markdown]
# # Define the external geometry
#
# The first step is to define the geometry of our problem, essentially a box on which we will apply some physical constraints and that will contain a set of materials. We can think of it as an "universe".
# The "laws" and their effects are calculated on a mesh, that mesh discretized our universe into finite elements.
#
# The geodynamics module allows you to quickly set up a model by creating a *Model* object.
# A series of arguments are required to define a *Model*:
#
#     - The number of elements in each direction elementRes=(nx, ny);
#     - The minimum coordinates for each axis minCoord=(minX, minY)
#     - The maximum coordinates for each axis maxCoord=(maxX, maxY)
#     - A vector that defines the magnitude and direction of the gravity components gravity=(gx, gy)

# %%
Model = GEO.Model(elementRes=(x_res, y_res),
                  minCoord=(0. * u.kilometer, -1.*Depth_of_box),
                  maxCoord=(model_length, Sticky_air * u.kilometer),
                  gravity=(0.0, -9.81 * u.meter / u.second**2))


# %% [markdown]
# Set output location to store data

# %%
today = datetime.now(pytz.timezone('Australia/Melbourne'))
if restart == False:
    Model.outputDir = os.path.join(model_velProfile.casefold() + f'-{(x_box/x_res)}_res-2000km_conv-{int(crust_dens.magnitude)}Dens-{int(D_surface.magnitude)}D-{int(crust_IH.magnitude)}IH'  + today.strftime('%Y-%m-%d_%H-%M') + "/")
    
    directory = os.getcwd() +'/'+ Model.outputDir

if restart == True:
    RestartDirectory = os.getcwd()
    directory = RestartDirectory


# %%
""" Additional swarm variables """
swarmStrainRateInv = Model.add_swarm_variable(name='SR_swarm', restart_variable=True, count=1)
# swarmViscousDissipation = Model.add_swarm_variable(name='VD_swarm', restart_variable=True, count=1)


""" Additional mesh variables """

meshViscousDissipation = Model.add_mesh_variable(name='VD_mesh', nodeDofCount=1 )

# %% [markdown]
# #### Add some Materials
#
# A material (or a phase) is first defined by the space it takes in the box (its shape).

# %%
air = Model.add_material(name="Air", shape=GEO.shapes.Layer(top=Model.top, bottom=0. * u.kilometer))


# %%
### Add materials
crust1 = Model.add_material(name="Crust1")
crust2 = Model.add_material(name="Crust2")
crust3 = Model.add_material(name="Crust3")
crust4 = Model.add_material(name="Crust4")

crust5 = Model.add_material(name="Crust5")
crust6 = Model.add_material(name="Crust6")

Sediment = Model.add_material(name="Sediment")

FarMantleLithosphere = Model.add_material(name="FarMantleLithosphere")

# %%
### Determine WZ locations using trig
Fault_positionX = crust_transition * u.kilometer


### Create a layered crust using some numpy manipulation
sin_function = np.sign(np.sin(GEO.dimensionalise(Model.swarm.data[:,1], u.kilometer)/(1.6 * u.kilometer)))

Model.materialField.data[(sin_function>0) & (Model.swarm.data[:,0] < GEO.nd(Fault_positionX+15.*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust1.index
Model.materialField.data[(sin_function<0) & (Model.swarm.data[:,0] < GEO.nd(Fault_positionX+15.*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust2.index

Model.materialField.data[(sin_function>0) & (Model.swarm.data[:,0] >= GEO.nd(Fault_positionX+15.*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust3.index
Model.materialField.data[(sin_function<0) & (Model.swarm.data[:,0] >= GEO.nd(Fault_positionX+15.*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust4.index



# %%
mantleLithosphere = Model.add_material(name="MantleLithosphere", shape=GEO.shapes.Layer(top=-1 * crustalthickness * u.kilometer, bottom=-97.5 * u.kilometer))

### fault angle, in degrees
angle_D = 45
Fault_PositionX_LAB = Fault_positionX + ((mantleLithosphere.top - mantleLithosphere.bottom) * math.tan(math.radians(angle_D)))


Strong_mantleLithosphere = Model.add_material(name="Strong_MantleLithosphere", shape=GEO.shapes.Polygon(vertices=[(Fault_positionX + 30.* u.kilometer, mantleLithosphere.top),
                                                                                                                    (Fault_PositionX_LAB+3.* u.kilometer, mantleLithosphere.bottom),
                                                                                                                    (Model.length, mantleLithosphere.bottom),
                                                                                                                    (Model.length, mantleLithosphere.top)]))

### Additional material to make far wall backstop in crust and mantle

Model.materialField.data[(sin_function>0) & (Model.swarm.data[:,0] >= GEO.nd(back_stop_x_coord*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust5.index
Model.materialField.data[(sin_function<0) & (Model.swarm.data[:,0] >= GEO.nd(back_stop_x_coord*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer))] = crust6.index
Model.materialField.data[(Model.swarm.data[:,0] >= GEO.nd(back_stop_x_coord*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(-1 * crustalthickness * u.kilometer)) &  (Model.swarm.data[:,1] > GEO.nd(-97.5*u.kilometer))] = FarMantleLithosphere.index


mantle = Model.add_material(name="Mantle", shape=GEO.shapes.Layer(top=mantleLithosphere.bottom, bottom=Model.bottom))




# %%
FaultShape = GEO.shapes.Polygon(vertices=[(Fault_positionX, mantleLithosphere.top),
                                        (Fault_positionX + 30.* u.kilometer, mantleLithosphere.top),
                                        (Fault_PositionX_LAB+3.* u.kilometer, mantleLithosphere.bottom),
                                         (Fault_PositionX_LAB, mantleLithosphere.bottom)])

Fault = Model.add_material(name="Fault", shape=FaultShape)


# %%
def Update_Material_LHS():
    sin_function = np.sign(np.sin(GEO.dimensionalise(Model.swarm.data[:,1], u.kilometer)/(1.6 * u.kilometer)))
    Model.materialField.data[(Model.swarm.data[:,0] < GEO.nd(Update_material_LHS_Length*u.kilometer)) &  (Model.swarm.data[:,1] > GEO.nd(0.*u.kilometer)) ] = air.index
    Model.materialField.data[(sin_function>0) & (Model.swarm.data[:,0] < GEO.nd(Update_material_LHS_Length*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer)) &  (Model.swarm.data[:,1] >= GEO.nd(-1.*crustalthickness*u.kilometer))] = crust1.index
    Model.materialField.data[(sin_function<0) & (Model.swarm.data[:,0] < GEO.nd(Update_material_LHS_Length*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(0.*u.kilometer)) &  (Model.swarm.data[:,1] >= GEO.nd(-1.*crustalthickness*u.kilometer))] = crust2.index
    Model.materialField.data[(Model.swarm.data[:,0] < GEO.nd(Update_material_LHS_Length*u.kilometer)) &  (Model.swarm.data[:,1] < GEO.nd(-1.*crustalthickness*u.kilometer)) &  (Model.swarm.data[:,1] >= GEO.nd(mantleLithosphere.bottom))] = mantleLithosphere.index


# %%
if uw.mpi.size == 1:
    Fig = visualisation.Figure(figsize=(1200,400), title="Material Field", quality=0)
    Fig.Points(Model.swarm, Model.materialField, fn_size=2.0)
    Fig.show()

# %% [markdown]
# ### Add material properties
#
# Start giving materials their properties

# %%
### Set global properties
Model.diffusivity = 1e-6 * u.metre**2 / u.second
Model.capacity    = 1000. * u.joule / (u.kelvin * u.kilogram)

# %%
#### set material properties
air.capacity = 100. * u.joule / (u.kelvin * u.kilogram)


# %%
### Set heat produciton properties

crust1.radiogenicHeatProd                   = crust_IH
crust2.radiogenicHeatProd                   = crust_IH
crust3.radiogenicHeatProd                   = crust_IH
crust4.radiogenicHeatProd                   = crust_IH
crust5.radiogenicHeatProd                   = crust_IH
crust6.radiogenicHeatProd                   = crust_IH

Sediment.radiogenicHeatProd                 = sediment_IH

mantleLithosphere.radiogenicHeatProd        = mantle_IH
mantle.radiogenicHeatProd                   = mantle_IH

Strong_mantleLithosphere.radiogenicHeatProd = mantle_IH
FarMantleLithosphere.radiogenicHeatProd     = mantle_IH

Fault.radiogenicHeatProd                    = mantle_IH

# %%
### Set density properties

air.density = 1. * u.kilogram / u.metre**3

### incoming plate
crust1.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)
crust2.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)
### middle plate
crust3.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)
crust4.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)
### Far plate
crust5.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)
crust6.density = GEO.LinearDensity(crust_dens, thermalExpansivity=3e-5 / u.kelvin)

Sediment.density = GEO.LinearDensity(sed_dens, thermalExpansivity=3e-5 / u.kelvin)



mantleLithosphere.density        = GEO.LinearDensity(mantle_dens, thermalExpansivity=3e-5 / u.kelvin)
mantle.density                   = GEO.LinearDensity(mantle_dens, thermalExpansivity=3e-5 / u.kelvin)
Strong_mantleLithosphere.density = GEO.LinearDensity(mantle_dens, thermalExpansivity=3e-5 / u.kelvin)
FarMantleLithosphere.density     = GEO.LinearDensity(mantle_dens, thermalExpansivity=3e-5 / u.kelvin)
Fault.density                    = GEO.LinearDensity(mantle_dens, thermalExpansivity=3e-5 / u.kelvin)

# %% [markdown]
#
#
#
#
# ###  Define Viscosities
#
# The rheology library *[GEO.ViscousCreepRegistry()]* contains some commonly used rheologies stored in a python dictionary structure.
#
# You can use dir(rh) to list all values in the UWGeo dictionary

# %%
rh = GEO.ViscousCreepRegistry()


# %%
### various rheologies that can be used

# ### Weak Zone Rheology
#rh.Wet_Olivine_Dislocation_Hirth_and_Kohlstedt_2003
# rh.Wet_Olivine_Diffusion_Hirth_and_Kohlstedt_2003
# rh.Wet_Olivine_Dislocation_Karato_and_Wu_1993

# ### Mantle Rheology
# rh.Dry_Olivine_Dislocation_Karato_and_Wu_1993
# rh.Dry_Olivine_Diffusion_Hirth_and_Kohlstedt_2003
# rh.Dry_Olivine_Dislocation_Hirth_and_Kohlstedt_2003

# ### Strong Crust
# rh.Dry_Quartz_Dislocation_Koch_et_al_1983
# rh.Dry_Maryland_Diabase_Dislocation_Mackwell_et_al_1998

# ### Weak Crust
#rh.Wet_Quartz_Dislocation_Tullis_et_al_2002
#rh.Wet_Quartz_Dislocation_Paterson_and_Luan_1990

#rh.Wet_Anorthite_Dislocation_Ribacki_et_al_2000

# %% [markdown]
# Define the viscous rheology for each of the materials

# %%

combined_viscosity_mantle = GEO.CompositeViscosity([rh.Dry_Olivine_Diffusion_Hirth_and_Kohlstedt_2003, rh.Dry_Olivine_Dislocation_Hirth_and_Kohlstedt_2003])
combined_viscosity_mantle_strong = GEO.CompositeViscosity([rh.Dry_Olivine_Diffusion_Hirth_and_Kohlstedt_2003, rh.Dry_Olivine_Dislocation_Hirth_and_Kohlstedt_2003])


combined_viscosity_fault = GEO.CompositeViscosity([rh.Wet_Olivine_Diffusion_Hirth_and_Kohlstedt_2003, rh.Wet_Olivine_Dislocation_Hirth_and_Kohlstedt_2003])


### Strong crust from Ranelli 1995
Diabase_Dislocation_Ranalli_1995 = GEO.ViscousCreep(preExponentialFactor=2.0e-4/u.megapascal**3.4/u.second,
                                                      stressExponent=3.4,
                                                      activationVolume=0.,
                                                      activationEnergy=260 * u.kilojoules/u.mole,
                                                      waterFugacity=0.0,
                                                      grainSize=0.0,
                                                      meltFraction=0.,
                                                      grainSizeExponent=0.,
                                                      waterFugacityExponent=0.,
                                                      meltFractionFactor=0.0,
                                                      f=1.0)

Quartzite_Dislocation_Ranalli_1995 = GEO.ViscousCreep(preExponentialFactor=6.7e-6/u.megapascal**2.4/u.second,
                                                      stressExponent=2.4,
                                                      activationVolume=0.,
                                                      activationEnergy=156 * u.kilojoules/u.mole,
                                                      waterFugacity=0.0,
                                                      grainSize=0.0,
                                                      meltFraction=0.,
                                                      grainSizeExponent=0.,
                                                      waterFugacityExponent=0.,
                                                      meltFractionFactor=0.0,
                                                      f=1.0)



# %%
Model.minViscosity = 1e19 * u.pascal * u.second
Model.maxViscosity = 1e24 * u.pascal * u.second

air.viscosity                = 1e19 * u.pascal * u.second


mantleLithosphere.viscosity        = combined_viscosity_mantle
mantle.viscosity                   = combined_viscosity_mantle

Strong_mantleLithosphere.viscosity = combined_viscosity_mantle
FarMantleLithosphere.viscosity     = combined_viscosity_mantle

Fault.viscosity                    = combined_viscosity_fault


# %%
## Lower plate (near left wall)
crust1.viscosity              = Quartzite_Dislocation_Ranalli_1995 # rh.Wet_Quartz_Dislocation_Tullis_et_al_2002 # rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012 # rh.Dry_Quartz_Dislocation_Koch_et_al_1983 #
crust2.viscosity              = Quartzite_Dislocation_Ranalli_1995 # rh.Wet_Quartz_Dislocation_Tullis_et_al_2002 # rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012 # rh.Dry_Quartz_Dislocation_Koch_et_al_1983 #

### upper plate (middle of box)
crust3.viscosity              = Quartzite_Dislocation_Ranalli_1995 # rh.Wet_Quartz_Dislocation_Tullis_et_al_2002 # rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012 # rh.Dry_Quartz_Dislocation_Koch_et_al_1983 #
crust4.viscosity              = Quartzite_Dislocation_Ranalli_1995 # rh.Wet_Quartz_Dislocation_Tullis_et_al_2002 # rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012 # rh.Dry_Quartz_Dislocation_Koch_et_al_1983 # 

### far plate (near right wall)
crust5.viscosity              = Diabase_Dislocation_Ranalli_1995
crust6.viscosity              = Diabase_Dislocation_Ranalli_1995

### Sediment
Sediment.viscosity            = 0.5 * Quartzite_Dislocation_Ranalli_1995 # rh.Wet_Quartz_Dislocation_Tullis_et_al_2002 # 

# %% [markdown]
# ### View temp and visc profile
# Works in serial only

# %% [markdown]
# ### Define Plasticity
#
# Plastic behavior is assigned using the same approach as for viscosities.

# %%
pl = GEO.PlasticityRegistry()


# %%
Sediment_plasticity = GEO.DruckerPrager(cohesion=10.* u.megapascal,
                                     cohesionAfterSoftening=1.*u.megapascal,
                                     frictionCoefficient=0.2,
                                     frictionAfterSoftening=0.1,
                                     epsilon1=0.5, epsilon2=1.5)

crust_plasticity = GEO.DruckerPrager(cohesion=10.* u.megapascal,
                                     cohesionAfterSoftening=1.*u.megapascal,
                                     frictionCoefficient=0.3,
                                     frictionAfterSoftening=0.15,
                                     epsilon1=0.5, epsilon2=1.5)

Strong_crust_plasticity = GEO.DruckerPrager(cohesion=10.* u.megapascal,
                                     cohesionAfterSoftening=1.*u.megapascal,
                                     frictionCoefficient=0.4,
                                     frictionAfterSoftening=0.2,
                                     epsilon1=0.5, epsilon2=1.5)

Mantle_plasticity =  GEO.DruckerPrager(cohesion=10.* u.megapascal,
                                     cohesionAfterSoftening=10.*u.megapascal,
                                     frictionCoefficient=0.6,
                                     frictionAfterSoftening=0.6,
                                     epsilon1=0.5, epsilon2=1.5)

Fault_plasticity = GEO.DruckerPrager(cohesion=10.* u.megapascal,
                                     cohesionAfterSoftening=1.*u.megapascal,
                                     frictionCoefficient=0.1,
                                     frictionAfterSoftening=0.05,
                                     epsilon1=0.5, epsilon2=1.5)


# %%


mantleLithosphere.plasticity        = Mantle_plasticity
mantle.plasticity                   = Mantle_plasticity
Strong_mantleLithosphere.plasticity = Mantle_plasticity
FarMantleLithosphere.plasticity     = Mantle_plasticity

Fault.plasticity                    = Fault_plasticity

# %%
### lower plate
crust1.plasticity              = crust_plasticity
crust2.plasticity              = crust_plasticity
### upper plate
crust3.plasticity              = crust_plasticity
crust4.plasticity              = crust_plasticity
### Far crust near back wall
crust5.plasticity              = crust_plasticity
crust6.plasticity              = crust_plasticity

#### sediment plasticity
Sediment.plasticity            = Sediment_plasticity



# ## Temperature Boundary Conditions

# %%
Model.set_temperatureBCs(top=273.15 * u.degK,
                         # bottom=1573.15 * u.degK,
                         materials=[(air, 273.15 * u.degK) ])#, (mantle, 1573.15* u.degK)])


# %%
# ## Velocity Boundary Conditions
Model.velocityField.data[:] = 0.


# %%
def updateVelocityBC(velocity):

    conv_vel = velocity * (u.centimeter / u.year)

    ### inflow/outflow across sticky air layer and constant vel across crust and lithospheric mantle
    conditionsA = [(Model.y < GEO.nd(0. * u.kilometre), GEO.nd(conv_vel)),
                   (True, GEO.nd(conv_vel) + Model.y * (GEO.nd((-2. * conv_vel) / GEO.nd(Sticky_air * u.kilometer))))]


    Left_wall_vel_top_changed = fn.branching.conditional(conditionsA)

    
    conditionsB = [(Model.y > GEO.nd(mantleLithosphere.bottom), Left_wall_vel_top_changed),
                   (True, (GEO.nd(conv_vel) + (Model.y-GEO.nd(mantleLithosphere.bottom)) * (GEO.nd(conv_vel) / GEO.nd(Depth_of_box+mantleLithosphere.bottom))))]


    Left_wall_vel_changed = fn.branching.conditional(conditionsB)

    #### decreasing velocity between the bottom of the lithosphere and base of the model
    Model.set_velocityBCs(left = [Left_wall_vel_changed, None],
                      right=[0., 0.],
                      top = [None, 0.])

# %%
if model_velProfile.casefold() == 'slow':
    vel = 2. ### cm/yr
    modelDuration = (totalConvergence*1e5 / vel)/1e6 #### time in Myr
    
    updateVelocityBC(velocity = vel)
    
    The_Checkpoint_interval = modelDuration / noOfTS
    

    
elif model_velProfile.casefold() == 'fast':
    vel = 10.
    modelDuration = (totalConvergence*1e5 / vel)/1e6 #### time in Myr
    
    updateVelocityBC(velocity = vel)

    The_Checkpoint_interval = modelDuration / noOfTS

    

else:
    modelDuration = 50. #### time in Myr
    
    The_Checkpoint_interval = modelDuration / noOfTS
    
    def decreasingVelFunction():
        vel = 12*(1-.059)**float(Model.time.magnitude/1e6)
        vel = np.where(vel >= 10, 10, vel)
        vel = np.where(vel <= 2, 2, vel)
        
        updateVelocityBC(velocity = vel)
        
    decreasingVelFunction()
    
    #### update velocity every loop
    Model.pre_solve_functions["updateVel"] = decreasingVelFunction
        
        
    
if uw.mpi.size == 1:
    print(f'model duration: {modelDuration} Myr')
    print(f'timestep interval: {The_Checkpoint_interval}')
    print(f'number of TS: {noOfTS}')


# %%
### Grid Tracers for the model
coords_circ = GEO.circles_grid(radius=5.0*u.kilometer,
                    minCoord=[Model.minCoord[0], mantleLithosphere.top],
                    maxCoord=[Model.maxCoord[0], air.bottom])


Model.add_passive_tracers(name="FSE_Crust", vertices=coords_circ)

### track pressure field, although may not be the best variable for thermodynamic evolution...
Model.FSE_Crust_tracers.add_tracked_field(Model.pressureField,
                              name="tracers_press",
                              units=u.megapascal,
                              dataType="double")

Model.FSE_Crust_tracers.add_tracked_field(Model.temperature,
                              name="tracers_temp",
                              units=u.degK,
                              dataType="double")

Model.FSE_Crust_tracers.add_tracked_field(Model.strainRate_2ndInvariant,
                              name="tracers_SR",
                                units=1./u.second,
                              dataType="double")

# %%
Model.init_model()


# %%
### Custom temperature gradient

# ### Loop version
# for index, coord in enumerate(Model.mesh.data):
# ### Temperature in air
#     if coord[1] > 0.:
#         T = (273.15 * u.kelvin)
#     #### Temperature across crust
#     elif coord[1] < 0. and coord[1] >= GEO.nd(-10*u.kilometer):
#             T = (273.15 * u.kelvin + (-1*GEO.dimensionalise(coord[1], u.kilometer) * 25. * u.kelvin/u.kilometer))
#     #### Temperature for the mantle lithosphere
#     elif coord[1] < GEO.nd(-10*u.kilometer) and coord[1] >= GEO.nd(mantleLithosphere.bottom):
#             T = ((273.15+130.0) * u.kelvin + (-1*GEO.dimensionalise(coord[1], u.kilometer) * 12. * u.kelvin/u.kilometer))
# #### Temperature for the Upper Mantle
#     elif coord[1] < GEO.nd(mantleLithosphere.bottom):
#         T = 1573.15* u.degK #(1573.15 * u.kelvin + (-1*GEO.dimensionalise(coord[1], u.kilometer) * 0.5 * u.kelvin/u.kilometer))

#     Model.temperature.data[index] = GEO.nd(T)
    
### numpy array manipulation version
Model.temperature.data[:] = 0.

Model.temperature.data[:,0][Model.mesh.data[:,1] > GEO.nd(air.bottom) ] = GEO.nd(273.15 * u.kelvin)

Model.temperature.data[:,0][(Model.mesh.data[:,1] < GEO.nd(air.bottom)) &  (Model.mesh.data[:,1] >= GEO.nd(-10*u.kilometer))] = GEO.nd(273.15 * u.kelvin + (-1*GEO.dimensionalise(Model.mesh.data[:,1][(Model.mesh.data[:,1] < GEO.nd(air.bottom)) &  (Model.mesh.data[:,1] >= GEO.nd(-10*u.kilometer))], u.kilometer) * 25. * u.kelvin/u.kilometer))

Model.temperature.data[:,0][(Model.mesh.data[:,1] < GEO.nd(-10*u.kilometer)) &  (Model.mesh.data[:,1] >= GEO.nd(mantleLithosphere.bottom))] = GEO.nd((273.15+130.0) * u.kelvin + (-1*GEO.dimensionalise(Model.mesh.data[:,1][(Model.mesh.data[:,1] < GEO.nd(-10*u.kilometer)) &  (Model.mesh.data[:,1] >= GEO.nd(mantleLithosphere.bottom))], u.kilometer) * 12. * u.kelvin/u.kilometer))

Model.temperature.data[:,0][(Model.mesh.data[:,1] < GEO.nd(mantleLithosphere.bottom))] = GEO.nd(1573.15* u.degK)



# %%
if uw.mpi.size == 0:
    dir(rh)

# %%
# Only run this when in serial. Will fail in parallel
if GEO.nProcs == 1:

    import matplotlib.pyplot as plt
    
    y = np.linspace(GEO.nd(air.top), GEO.nd(mantleLithosphere.bottom), 1000)
    x = np.zeros_like(y)+GEO.nd(100*u.kilometer)
    
    coords = np.ndarray((y.shape[0], 2))
    
    coords[:,0], coords[:,1] = x, y
    
    # print(coords)
    
    temp = Model.temperature.evaluate(coords)
    
    ### too weak
    crust1.viscosity = 1.*Quartzite_Dislocation_Ranalli_1995
    crust2.viscosity = 1.*Quartzite_Dislocation_Ranalli_1995
    visc_RQ = Model.viscosityField.evaluate(coords) # rh.Dry_Mafic_Granulite_Dislocation_Wang_et_al_2012
    

    Fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,7), sharey=True)
    ax1.plot(GEO.dimensionalise(temp, u.degK), GEO.dimensionalise(y, u.kilometer))
    ax1.set_xlabel("Temperature in Kelvin")
    ax1.set_ylabel("Depth in kms")
    ax1.set_ylim(-100, 40)
    ax1.set_title("Temperature profile")


    ax2.plot((GEO.dimensionalise(visc_RQ, u.pascal * u.second)), GEO.dimensionalise(y, u.kilometer), ls=":")
    ax2.set_xlabel("Viscosity in Pa S")
    ax2.set_ylabel("Depth in kms")
    ax2.set_title("Viscosity profile")
    ax2.set_ylim(-40, 40)
    plt.show()


# %%
def update_custom_fields():

    # swarmViscousDissipation.data[:] = 2. * GEO.dimensionalise(Model.strainRate_2ndInvariant.evaluate(Model.swarm), 1./u.second) * GEO.dimensionalise(Model.strainRate_2ndInvariant.evaluate(Model.swarm), 1./u.second) * GEO.dimensionalise(Model.viscosityField.data[:], u.pascal * u.second)
    ### Calculation on mesh
    #Model.meshViscousDissipation.data[:] = 2. * Model.strainRate_2ndInvariant.evaluate( Model.mesh) * Model.strainRate_2ndInvariant.evaluate( Model.mesh) * Model.viscosityField.evaluate( Model.mesh)
    meshViscousDissipation.data[:] = 2. * (GEO.dimensionalise(Model.strainRate_2ndInvariant.evaluate(Model.mesh), 1./u.second) * GEO.dimensionalise(Model.strainRate_2ndInvariant.evaluate(Model.mesh), 1./u.second) * GEO.dimensionalise(Model.projViscosityField.data[:], u.pascal * u.second)).magnitude

    swarmStrainRateInv.data[:] = (GEO.dimensionalise(Model.strainRate_2ndInvariant.evaluate( Model.swarm), 1./u.second)).magnitude
    # viscositySwarm.data[:] = GEO.dimensionalise(Model.viscosityField.evaluate( Model.swarm), u.pascal * u.second)

    # print('mesh VD data: ', meshViscousDissipation.data[1:10])


### Calculate the energy dissipated in various areas of the model
VD_Ucrust = []
VD_Lcrust = []
VD_Bcrust = []
VD_Sed    = []
VD_model  = []



def viscous_dissipation_calc():
    VD_model_df = pd.DataFrame()
    c1 = Model.materialField > (crust1.index - 0.5)
    c2 = Model.materialField < (crust2.index + 0.5)

    lower_plate = c1 & c2

    c3 = Model.materialField > (crust3.index - 0.5)
    c4 = Model.materialField < (crust4.index + 0.5)

    upper_plate = c3 & c4

    c5 = Model.materialField > (crust5.index - 0.5)
    c6 = Model.materialField < (crust6.index + 0.5)

    back_plate = c5 & c6

    model_material = Model.materialField > (crust1.index - 0.5)

    s1 = Model.materialField > (Sediment.index - 0.5)
    s2 = Model.materialField < (Sediment.index + 0.5)

    sed = s1 & s2


    # how to calculate VD
    vd = 2. * Model._viscosityFn * Model.strainRate_2ndInvariant**2
    # vd_dimensionalised = GEO.dimensionalise(vd, u.pascal / u.second)

    """lower plate VD"""
    clause = [ (lower_plate, vd),
                ( True   , 0.) ]

    fn_crust_vd = fn.branching.conditional( clause )

    VD_Lcrust.append(2. * Model.mesh.integrate(fn_crust_vd)[0])
    """upper plate VD"""
    clause = [ (upper_plate, vd),
                ( True   , 0.) ]

    fn_crust_vd = fn.branching.conditional( clause )

    VD_Ucrust.append(2. * Model.mesh.integrate(fn_crust_vd)[0])
    """Back plate VD"""
    clause = [ (back_plate, vd),
                ( True   , 0.) ]

    fn_crust_vd = fn.branching.conditional( clause )

    VD_Bcrust.append(2. * Model.mesh.integrate(fn_crust_vd)[0])

    """Sediment VD"""
    clause = [ (sed, vd),
                ( True   , 0.) ]

    fn_crust_vd = fn.branching.conditional( clause )

    VD_Sed.append(2. * Model.mesh.integrate(fn_crust_vd)[0])


    '''Model VD'''

    clause = [ (model_material, vd),
                ( True   , 0.) ]

    fn_model_vd = fn.branching.conditional( clause )

    VD_model.append(2. * Model.mesh.integrate(fn_model_vd)[0])

    '''Save VD to DF'''
    VD_model_df['Lower Crust'] = VD_Lcrust
    VD_model_df['Upper Crust'] = VD_Ucrust
    VD_model_df['Back Crust']  = VD_Bcrust
    VD_model_df['Sediment']    = VD_Sed

    VD_model_df['Total']       = VD_model

    VD_model_df.to_csv(directory + 'VD_data.csv')




def CumulativeStrainCheck():
    Model.plasticStrain.data[Model.swarm.data[:,0]<(GEO.nd(x_threshold_Strain_weakening*u.kilometer))] = 0.

def Additional_files():

    file_prefix = os.path.join(directory, 'viscositySwarm-%s' % Model.checkpointID)
    handle = Model.viscosityField.save('%s.h5' % file_prefix)

def Checkpoint_additional_stuff():
    # global TotalConvergence
    # TotalConvergence += GEO.dimensionalise((GEO.nd((conv_vel* u.centimeter / u.year) * GEO.dimensionalise(Model._dt, u.megayear))), u.kilometer)


    ### Stops strain on new incoming materials

    # CumulativeStrainCheck()

    if GEO.nd(round(Model.time,0) % round(The_Checkpoint_interval * 1e6 * u.years, 0)) == 0.:
        Additional_files()
        viscous_dissipation_calc()



    # if GEO.nd(round(Model.time,0) % round(2.*The_Checkpoint_interval * 1e6 * u.years, 0)) == 0.:
# Additional_files()
# viscous_dissipation_calc()
update_custom_fields()
# Checkpoint_additional_stuff()


# %%
### Add new variables to output
GEO.rcParams["default.outputs"].append("SR_swarm")
GEO.rcParams["default.outputs"].append("VD_mesh")
GEO.rcParams["default.outputs"].append("viscosityField")

### remove variables from output
GEO.rcParams["default.outputs"].remove("projMeltField")
GEO.rcParams["default.outputs"].remove("projDensityField")
GEO.rcParams["default.outputs"].remove("projPlasticStrain")


# %%
# ### if superlu (local) /superludist (HPC)/mumps (local/HPC), high penalty

### original settings
# Model.solver.set_inner_method("mg")
# Model.solver.set_penalty(1e-3)

### optimal settings (?)
Model.solver.set_inner_method("mumps")
Model.solver.options.scr.ksp_type="cg"
Model.solver.set_penalty(1.0e7)



### additional functions for the model

Model.pre_solve_functions["StrainCheck"] = CumulativeStrainCheck
Model.pre_solve_functions["updateMat"] = Update_Material_LHS
Model.pre_solve_functions["customFields"] = update_custom_fields
Model.pre_solve_functions["additionalFiles"] = Checkpoint_additional_stuff


### keep material updated LHS
Model.post_solve_functions["A-post"] = Update_Material_LHS



# %%
if D_surface.magnitude > 0.: 
    ### set up surface tracers every 1 km
    npoints = int(Model.length.magnitude)
    
    coords = np.ndarray((npoints, 2))
    coords[:, 0] = np.linspace(GEO.nd(Model.minCoord[0]), GEO.nd(Model.maxCoord[0]), npoints)
    coords[:, 1] = GEO.nd(air.bottom)
    
    Model.surfaceProcesses = GEO.surfaceProcesses.diffusiveSurface_2D(
    airIndex=air.index,
    sedimentIndex=Sediment.index,
    D= D_surface, ### diffusion value for surface, set at beginning of the script
    surfaceArray = coords,
    updateSurfaceLB = Update_material_LHS_Length * u.kilometer
)
    
    if uw.mpi.rank == 0:
        print('surface processes initialised')

    ### track the temperature at the surface nodes
    Model.surface_tracers.add_tracked_field(Model.temperature,
                                name="temperature",
                                units=u.degK,
                                dataType="float", count=1)
    
else:
    pass

# %%
# #### Run the Model
Model.run_for((modelDuration* u.megayears)+(0.0001 * u.megayears), checkpoint_interval=The_Checkpoint_interval*u.megayears)

# %%
