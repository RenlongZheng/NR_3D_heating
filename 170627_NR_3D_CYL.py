import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

#os.remove('Output.txt')
#text_file = open("Output.txt", "w")

# Set variables here
length = 20e-6                      # Length of the NR (m)
width = 5e-6                        # Width of the NR (m)
height = 52e-9                      # Height of the NR (m)
T0 = 77                             # Cryostat temperature (K)
Tinf = 298                          # Ambient temperature (K)
kappa = 40                          # Thermal conductivity (W/(m.K))
n_CdS = 2.5513                      # Refractive index real part (532 nm)
k_CdS = 0.0057657                   # Refractive index imag part (532 nm)
#n_CdS = 2.6399                     # Refractive index real part (514 nm)
#k_CdS = 0.055754                   # Refractive index imag part (514 nm)
ref = 0.19                          # Reflectivity (532 nm)
#ref = 0.20317                      # Reflectivity (514 nm)
spotDiameter = 2.5e-6               # Laser spot diameter (m)
w0 = spotDiameter/2                 # Laser spot radius (m)
sampleRate = 50                     # Spatial sample rate
numSum = 50                         # Eigenvalue sample rate
lambda0 = 532e-9                    # laser wavelength (m)
Power = 0.005*(1-ref)               # Laser power (W)
spotPos = 0.75                      # Normalized position of laser spot (0: base, 1: tip)

# Few calculations, using defined variables above.
r_max = np.sqrt(2.303*np.square(w0))
kpp = (2*math.pi*k_CdS)/(lambda0)
L0 = spotPos*length
I0 = (Power*2)/(math.pi*np.square(w0))

theta = np.linspace(0, math.pi, sampleRate)
r = np.linspace(0, r_max, sampleRate)
u = (2*np.square(r))/(np.square(w0))
a = length/width
b = length/height
l = np.linspace(0, numSum, numSum+1)
m = np.linspace(0, numSum, numSum+1)
n = np.linspace(0, numSum, numSum+1)

def xEigFun(r,theta,l_star):
    if l_star == 0:
        return 1/np.sqrt(2)
    else:
        return np.cos(l_star*((width/2)+(r*np.sin(theta))))

def yEigFun(r,theta,m_star):
        return np.sin(m_star*(L0-(r*np.cos(theta))))

def zEigFun(zeta,n):
    if n == 0:
        return 1
    else:
        return np.sqrt(2)*np.cos(n*math.pi*zeta)

# Eigen fuctions that can take vector inputs
xfunc = np.vectorize(xEigFun, otypes=[np.float])
yfunc = np.vectorize(yEigFun, otypes=[np.float])
zfunc = np.vectorize(zEigFun, otypes=[np.float])

# Funtion to Calculate the double integral for respective 'l' and 'm' values.
# Optional output: plot of first integral vs. u
def double_integral(ll,mm):
    integral_xy = np.zeros(sampleRate)
    l_star = 2 * math.pi * ll / width
    m_star = (mm + 1 / 2) * math.pi / length
    for i in range(0,sampleRate):
        integral_xy[i] = (8/(width*length))*np.trapz(np.exp(-2*np.square(r[i])/np.square(w0))*xEigFun(r[i],theta,l_star)* yEigFun(r[i],theta,m_star)*r[i], x=theta)   # Calculation of the first integral
        #print(integral_xy[i])
    #plt.plot(r,integral_xy)                 # Plot of first integral vs. u
    #plt.show()
    return(np.trapz(integral_xy,r))         # Calculation and return of 2nd integral

# Enter values for l and m as "double_integral(l,m)" to obtain double integral and plot of first integral vs. u
#print(double_integral(1,0))
def gammaEigSquared(ll,mm,nn):
    return (np.square(a*ll*math.pi) + np.square((mm+0.5)*math.pi) + np.square(b*nn*math.pi))
#print(gammaEigSquared(2,2,2))

# Function of Z_analytical
def z_ana(nn):
    if nn == 0:
        return (1 - np.exp(-2 * kpp * height)) / (2 * kpp * height)
    else:
        return ((np.sqrt(2)*2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*np.cos(nn*math.pi))

# Function to get A_lmn for respective values of l, m and n
def A_lmn(ll,mm,nn):
    return ((4/gammaEigSquared(ll,mm,nn))*double_integral(ll,mm)*z_ana(nn))
#print(A_lmn(1,2,3))

triple_sum_vals = np.zeros((numSum,numSum,numSum))
def U(xi, eta, zeta):
    for mm in range(0, numSum):
        for nn in range(0, numSum):
            for ll in range(0, numSum):
                triple_sum_vals[ll,mm,nn] = A_lmn(ll,mm,nn)*xfunc(xi,ll)*yfunc(eta,mm)*zfunc(zeta,nn)
    triple_sum = np.sum(np.sum(np.sum(triple_sum_vals)))
    return ((((triple_sum)+((T0-Tinf)/Tinf))*Tinf)+Tinf)
#print("Temp difference at the spot: ", U(0.5, L0/length, 0)-T0, " K")

#Loop for the 2D line plot.
plot_rate = 20                                   # Plotting resolution. Only these values will be calculated
temp = np.zeros(plot_rate)                       # Defining 0 temperature vector for filling in the loop below
for i in range(0,plot_rate):
    temp[i] = (U(0.5,i/plot_rate,0))
    print(temp[i])                               # Printing calculated temperature values.
plt.plot(temp)
plt.show()
