'''
This is a code for calculating the Temperature gradient on the surface of a cantilever, irradiated with a Gaussian
laser spot at a specified position on the surface centerline.
Based on theory by James Davis - Refer to Fiber Heating 6.0.docx

Author: Anupum Pant
Date: June 13, 2017
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

#os.remove('Output.txt')
#text_file = open("Output.txt", "w")

# Set variables here
length = 25e-6                      # Length of the NR (m)
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
sampleRate = 10                     # Spatial sample rate
numSum = 10                         # Eigenvalue sample rate
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

# Eigen functions. X, Y and Z
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

# Conversion of Eigen fuctions for accepting vector inputs
xfunc = np.vectorize(xEigFun, otypes=[np.float])
yfunc = np.vectorize(yEigFun, otypes=[np.float])
zfunc = np.vectorize(zEigFun, otypes=[np.float])

def sigma_xy(i):
    return ((2*(w0**2/(4*width*length))*np.square(length)*kpp)/(kappa*Tinf))*I0*np.exp(-u[i])

# Funtion to Calculate the double integral for respective 'l' and 'm' values.
# Optional output: plot of first integral vs. u
def double_integral(ll,mm):
    integral_xy = np.zeros(sampleRate)
    for i in range(0,sampleRate):
        eta = ((L0/length)-((w0/length)*(np.sqrt(u[i]/2))*np.cos(theta)))
        xi = (0.5)+((np.sqrt(u[i]/2))*(w0/width)*(np.sin(theta)))
        integral_xy[i] = 2*np.trapz((sigma_xy(i)*xfunc(xi,ll)*yfunc(eta,mm)), x=theta)    # Calculation of the first integral
    #plt.plot(u,integral_xy)                 # Plot of first integral vs. u
    #plt.show()
    return (np.trapz(integral_xy,x=u))         # Calculation and return of 2nd integral

# Enter values for l and m as "double_integral(l,m)" to obtain double integral and plot of first integral vs. u
#print(double_integral(0,1))

# Function to get gammaEigSquared for respective l, m and n values
def gammaEigSquared(ll,mm,nn):
    return (np.square(a*ll*math.pi) + np.square((mm+0.5)*math.pi) + np.square(b*nn*math.pi))
#print(gammaEigSquared(2,2,2))

# Function of Z_analytical
def z_ana(nn):
    if nn == 0:
        return ((2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*np.cos(nn*math.pi))
    else:
        return ((np.sqrt(2)*2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*np.cos(nn*math.pi))

# Function to get A_lmn for respective values of l, m and n
def A_lmn(ll,mm,nn):
    return ((4/gammaEigSquared(ll,mm,nn))*double_integral(ll,mm)*z_ana(nn))
#print(A_lmn(1,2,1))
A_lmn_matrix = np.zeros((numSum,numSum,numSum))
for mm in range(0, numSum):
    for nn in range(0, numSum):
        for ll in range(0, numSum):
            A_lmn_matrix[ll,mm,nn] = A_lmn(ll,mm,nn)
source_3D = np.sum(np.sum(np.sum(A_lmn_matrix)))
print(source_3D)
base_area = width*height
#print(base_area)
flux = 11067006.6229
print(40*flux*base_area)

