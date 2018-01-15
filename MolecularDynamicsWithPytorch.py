import numpy as np
import matplotlib.pyplot as plt
import torch
#------------------------------INPUT PARAMETERS-----------------------------------------------------------------
PlotOnScreen=True # Plot particles on screen during simulation
UseGPU=True # Use GPU if cuda available
NumSteps=10000 # number of simulation steps
TimeSteps=100 # Display particles once in TimeSteps
PeriodicBoundary=True# does the simulation use preiodic boundary if so make sure to set cell size paramter
CellSize= 1.2 # Size of cell in simulation important if you use priodic boundary conditions
dt=0.00005# Time step: time lapse of molecular dynamic simulation step (if this too big the simulation will explode if its too small it can take lots of time)
m=0.5# mass of particles
NumParticles=100 # Number of particles to generate
X = torch.rand(NumParticles,1)*CellSize# x position of particles  assume 2d system
Y  = torch.rand(NumParticles,1)*CellSize# y position of particles  assume 2d system
Vx = (torch.ones(NumParticles,1)*0.5-torch.rand(NumParticles,1))*0.0 # velocities on x axis of particles assume 2d system
Vy = (torch.ones(NumParticles,1)*0.5-torch.rand(NumParticles,1))*0.0 # velocities on x axis of particles assume 2d system
CoolingRate= 0.995# Increase/decrease the speed of all particles by factor to achive cooling/heating effect.
#---------------------------------If you use priodic boundary condition generate all neighbor cells---------------------------------------------------------------------------

#------------------------------------------------------------------------------------------
if PeriodicBoundary:
    NumParticlesAll = NumParticles*9  # num particles in cell and neighbour cells
    CellPos = torch.Tensor(9,2)  # Generate cordinates particles of all the neighbors cells (by tranlating the particles coordinates)
    f = 0
    for i in range(-1, 2):  # Generate cordinates of all the neighbors cells
        for i2 in range(-1, 2):
            CellPos[f, :] = torch.Tensor([i, i2])  # Generate the corner (0,0) position of each of the neighbouring cells
            f += 1
    Xcell = torch.mm(CellPos[:, 0].unsqueeze(1), torch.ones([1, NumParticles]))
    Ycell = torch.mm(CellPos[:, 1].unsqueeze(1), torch.ones([1, NumParticles]))
else:
    NumParticlesAll = NumParticles  # num particles
#-----------------Supporting tensor that will be used during the main loop----------------------------------------------------
ones9=torch.ones([1, 9])
OnesNumParicleAll=torch.ones([1, NumParticlesAll])
OnesNumParicles=torch.ones([NumParticles,1])
Rmin=torch.ones([NumParticles,NumParticlesAll]) * 0.02
#-----------------main simulation loop-------------------------------------------------------------------
# Use Cuda if CUDA is available and UseGPu is true, convert all torch tensor to cuda
if UseGPU and torch.cuda.is_available():
    if PeriodicBoundary:
         Xcell = Xcell.cuda()
         Ycell = Ycell.cuda()
         ones9 = ones9.cuda()

    X=X.cuda()
    Y=Y.cuda()
    Vx=Vx.cuda()
    Vy=Vy.cuda()
    OnesNumParicleAll=OnesNumParicleAll.cuda()
    OnesNumParicles=OnesNumParicles.cuda()
    Rmin=Rmin.cuda()
#-------------------------------Preparing to display on screen-------------------------------------------------------
if PlotOnScreen:
    plt.ion()  # make sure plot can be update
    fig = plt.figure()  # start plot
#------------------------------------------------------------------------------------------
for i in range(NumSteps):
    print("Step "+str(i))
    if PeriodicBoundary: # If priodic boundary Generate cordinates of all particles in all neighbor cells by replicating current cell
        Xall = torch.mm(X, ones9).transpose(1,0) + Xcell  # Generate postion of particles in all neighbor cells
        Yall = torch.mm(Y, ones9).transpose(1, 0) + Ycell
        Xall=Xall.view(1,-1) # Flatten to 1d Tensor
        Yall=Yall.view(1,-1) #
    else:
        Xall=X.transpose(1,0) # in none priodic boundary conditions assume one cell with infinite size
        Yall=Y.transpose(1,0)
 #-----------Calculate distance between every particle pairs  along X and y axis------------------------------------------------
    Rx = torch.mm(X, OnesNumParicleAll) - torch.mm(OnesNumParicles, Xall)
    Ry = torch.mm(Y, OnesNumParicleAll) - torch.mm(OnesNumParicles, Yall)

    R2 = Rx*Rx+Ry*Ry # The squre distance between every pair of particles
    R = torch.sqrt(R2) # The distance between every pair of particles

    R = torch.max(R, Rmin)  # To avoid division by zero make min distance larger then 0
    F = -30 / torch.pow(R, 2) + 10 / torch.pow(R, 3) # force between every pair of particles
    Fx=torch.sum(F*Rx/R,dim=1) # Total Force along x axis for each particle
    Fy=torch.sum(F*Ry/R,dim=1) # Total Force along y axis for each particle
    Ax=dt*Fx/m # Acceleration along x axis
    Ay=dt*Fy/m # Acceleration along y axis
    Vx+=Ax # Update particles speed
    Vy+=Ay
    Vx*=CoolingRate # reduce particle speed according to cooling rate
    Vy*=CoolingRate
    X += Vx * dt # Update particles locations
    Y += Vy * dt
    if PeriodicBoundary:
        X = torch.fmod(X + CellSize,CellSize)  # periodic boundary conditions make sure the particle never exit the cell
        Y = torch.fmod(Y + CellSize,CellSize)
#............Plot Particles on screen........................................
    if PlotOnScreen and i%TimeSteps==0: # Plot particles on screen
        plt.clf()  # clear figure
        plt.xlim(0, CellSize)  # define figure axis size
        plt.ylim(0, CellSize)  # define figure axis size
        plt.title(["step ", i])  # figure  title
        plt.scatter(X.cpu().numpy(),Y.cpu().numpy())  # add particle position to graph
        plt.show()  # show on screen
        plt.pause(0.001)  # time delay







