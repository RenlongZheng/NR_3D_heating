import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math

#Set variables here
length = 20e-6                      # Length of the NR (m)
width = 5e-6                        # Width of the NR (m)
height = 52e-9                      # Height of the NR (m)
T0 = 77                             # Cryostat temperature (K)
Tinf = 298                          # Anbient temperature (K)
kappa = 40                          # Thermal conductivity (W/(m.K))
n_CdS = 2.5513                      # Refractive index real part (532 nm)
k_CdS = 0.0057657                   # Refractive index imag part (532 nm)
#n_CdS = 2.6399                     # Refractive index real part (514 nm)
#k_CdS = 0.055754                   # Refractive index imag part (514 nm)
ref = 0.19                          # Reflectivity (532 nm)
#ref = 0.20317                      # Reflectivity (514 nm)
spotDiameter = 2.5e-6               # Laser spot diameter (m)
w0 = spotDiameter/2                 # Laser spot radius (m)
sampleRate = 40                   # Spatial sample rate
numSum = 10                         # Eigenvalue sample rate
lambda0 = 532e-9                    # laser wavelength (m)
Power = 0.005*(1-ref)               # Laser power (W)
spotPos = 0.75                      # Normalized position of laser spot (0-1) of L

#Few calculations, refer to notes
r_max = np.sqrt(2.30258509299*w0**2)
kpp = 2*math.pi*k_CdS/(lambda0)
L0 = spotPos*length
I0 = Power*2/(math.pi*np.square(w0))

theta = np.linspace(0, math.pi, sampleRate)
r = np.linspace(0, r_max, sampleRate)
u_max = 2*r_max**2/w0**2
u = np.linspace(0, u_max, sampleRate)
print(u)
l = np.linspace(0, numSum, numSum+1)
m = np.linspace(0, numSum, numSum+1)
n = np.linspace(0, numSum, numSum+1)
#print(2*r_max**2/w0**2)
# Eigen functions
def xEigFun(xi,l):
    if l == 0:
        return 1
    else:
        return np.sqrt(2)*np.cos(l*math.pi*xi)

def yEigFun(eta,m):
        return np.sqrt(2)*np.sin((m+0.5)*math.pi*eta)

def zEigFun(zeta,n):
    if n == 0:
        return 1
    else:
        return np.sqrt(2)*np.cos(n*math.pi*zeta)

# Eigen fuctions that can take vector inputs
xfunc = np.vectorize(xEigFun, otypes=[np.float])
yfunc = np.vectorize(yEigFun, otypes=[np.float])
zfunc = np.vectorize(zEigFun, otypes=[np.float])

integral_xy = np.zeros(sampleRate)
for i in range(0, sampleRate):
    eta = ((L0 / length) - ((w0 / length) * (np.sqrt(u[i] / 2)) * np.cos(theta)))
    xi = (0.5) + ((np.sqrt(u[i] / 2)) * (w0 / width) * (np.sin(theta)))
    sigma_xy = np.exp(-u[i])
    # print(xfunc(xi,ll))
    # print(yfunc(eta,mm))
    integral_xy[i] = np.trapz((sigma_xy * xfunc(xi, 1) * yfunc(eta, 0)), theta)  # Calculation of the first integral
    # print(integral_xy[i])
plt.plot(u, integral_xy)  # Plot of first integral vs. u
plt.show()