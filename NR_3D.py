import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

length = 25e-6
width = 5e-6
height = 52e-9
T0 = 298
Tinf = 298
kappa = 40
n_CdS = 2.5513
k_CdS = 0.005765
#n_CdS = 2.6399
#k_CdS = 0.055754
spotDiameter = 2.5e-6
sampleRate = 20
numSum = 20
lambda0 = 532e-9
kpp = 2*math.pi*k_CdS/(lambda0)
ref = 0.19
#ref = 0.20317
Power = 0.005*(1-ref)
spotPos = 0.30
yBeamCenter = spotPos*length
numCount = np.linspace(1,numSum,numSum)

xEigVals = (numCount-1)*math.pi
yEigVals = (numCount-0.5)*math.pi
zEigVals = xEigVals
a = length/width
b = length/height

xEigVals_rep = np.tile(np.tile(xEigVals,(numSum,1)),(numSum,1,1))
yEigVals_rep = np.tile(np.transpose(np.tile(yEigVals,(numSum,1))),(numSum,1,1))
zEigVals_rep = np.transpose(np.tile(np.tile(zEigVals,(numSum,1)),(numSum,1,1)),(2,1,0))
gammaEigSquared = np.square(a*xEigVals_rep) + np.square(yEigVals_rep) + np.square(b*zEigVals_rep)

x = np.linspace(0, 1, sampleRate)
y = np.linspace(0, 1, sampleRate)
z = np.linspace(0, 1, sampleRate)
x1 = np.linspace(0, 1, sampleRate+1)
y1 = np.linspace(0, 1, sampleRate+1)
z1 = np.linspace(0, 1, sampleRate+1)
etaVals, zetaVals, xiVals = np.meshgrid(x,y,z)
etaValsInt, zetaValsInt, xiValsInt = np.meshgrid(x1,y1,z1)

xVals = xiVals*width
yVals = etaVals*length
zVals = zetaVals*height

xValsInt = xiValsInt*width
yValsInt = etaValsInt*length
zValsInt = zetaValsInt*height

xVectInt = np.linspace(0, 1, sampleRate+1)
yVectInt = np.linspace(0, 1, sampleRate+1)
zVectInt = np.linspace(0, 1, sampleRate+1)

Irradiance = Power*2/(math.pi*np.square((spotDiameter/2)))
source = np.exp((-2*(np.square((xValsInt-width/2))+np.square((yValsInt-yBeamCenter))))/np.square((spotDiameter/2)))*4*math.pi*k_CdS/(lambda0)*Irradiance*np.exp(-2*kpp*zValsInt)
sigma = (source*np.square(length))/(Tinf*kappa)

AmnTrapz = np.zeros((numSum,numSum,numSum))

def xEigFun(xi,eig):
    if eig == 0:
        return 1
    else:
        return np.sqrt(2)*np.cos(eig*xi)

def yEigFun(eta,eig):
        return np.sqrt(2)*np.sin(eig*eta)

def zEigFun(xi,eig):
    if eig == 0:
        return 1
    else:
        return np.sqrt(2)*np.cos(eig*xi)

xfunc = np.vectorize(xEigFun, otypes=[np.float])
yfunc = np.vectorize(yEigFun, otypes=[np.float])
zfunc = np.vectorize(zEigFun, otypes=[np.float])

for mm in range(0,numSum):
    for nn in range(0,numSum):
        for ll in range(0,numSum):
            #print(xEigVals[nn])
            #print(xiValsInt)
            xVal = xfunc(xiValsInt, xEigVals[nn])
            #print(xVal)
            yVal = yfunc(etaValsInt, yEigVals[ll])
            zVal = zfunc(zetaValsInt, zEigVals[mm])
            tempVals = xVal*yVal*zVal*sigma
            AmnTrapz[mm,nn,ll] = (4/gammaEigSquared[mm,nn,ll])*np.trapz(np.trapz(np.trapz(tempVals,zVectInt,2),yVectInt,1),xVectInt,0)

#print(xVal[1,1,:])
#print(yVal[1,:,1])
#print(zVal[:,1,1])

tempResultTrapz = np.zeros((numSum,numSum,numSum))
resultTrapz = np.zeros((sampleRate,sampleRate,sampleRate))
for mm in range(0,sampleRate):
    for nn in range(0,sampleRate):
        for ll in range(0,sampleRate):
            a = xfunc(xiVals[ll,mm,nn], xEigVals).transpose()
            a.shape = (numSum,1)
            b = yfunc(etaVals[ll,mm,nn], yEigVals)
            b.shape = (1,numSum)
            tempResultTrapz2D = a.dot(b)
            zTempResult = np.transpose(np.tile(zfunc(zetaVals[ll,mm,nn],zEigVals),(numSum,numSum,1)),(2,0,1))
            for oo in range(0,numSum):
                tempResultTrapz[oo,:, :] = AmnTrapz[oo,:,:]*zTempResult[oo,:,:]*tempResultTrapz2D
            resultTrapz[ll,mm,nn] = ((T0-Tinf)/Tinf + np.sum(np.sum(np.sum(tempResultTrapz)))*Tinf+Tinf)

min_Temp = np.min(np.min(np.min(resultTrapz)))
max_Temp = np.max(np.max(np.max(resultTrapz)))
mean_Temp = np.mean(np.mean(resultTrapz[:,:,1]))

print("Maximum Temperature = ",max_Temp,"K")
print("Average Temperature = ",mean_Temp,"K")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = xVals[:,1,:]*10**6
y = yVals[:,:,1].transpose()*10**6
z = resultTrapz[1,:,:]

surf = ax.plot_surface(x, y, z, cmap=cm.jet, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.7, aspect=10)
ax.set_xlabel('Width ($\mu$m)', fontsize=10)
ax.set_ylabel('Length ($\mu$m)', fontsize=10)
ax.set_zlabel('Temperature (K)')

plt.show()









