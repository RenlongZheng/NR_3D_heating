'''
This is a code for calculating the Temperature gradient on the surface of a cantilever, irradiated with a Gaussian
laser spot at a specified position on the surface centerline.
Based on theory by James Davis - Refer to Fiber Heating 6.0.docx

Author: Anupum Pant
Date: June 13, 2017
Updated: July 6, 2017
'''

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

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
sampleRate = 30                     # Spatial sample rate
numSum = 10                         # Eigenvalue sample rate
lambda0 = 532e-9                    # laser wavelength (m)
laser_power = 5                     # Laser power in milliwatts
Power = laser_power*10**-3*(1-ref)               # Laser power (W)
spotPos = 0.75                      # Normalized position of laser spot (0: base, 1: tip)

# Few calculations, using defined variables above.
r_max = np.sqrt(2.303*np.square(w0))
kpp = (2*math.pi*k_CdS)/(lambda0)
L0 = spotPos*length
I0 = (Power*2)/(math.pi*np.square(w0))

theta = np.linspace(0, math.pi, sampleRate)
r = np.linspace(0, r_max, sampleRate)
#u = (2*np.square(r))/(np.square(w0))
u = np.linspace(0, 8, sampleRate)
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

# Funtion to Calculate the double integral for respective 'l' and 'm' values.
# Optional output: plot of first integral1 vs. u
def double_integral(ll,mm):
    integral_xy = np.zeros(sampleRate)
    for i in range(0,sampleRate):
        sigma_xy = ((w0**2*length*kpp)/(width*kappa*T0))*I0*np.exp(-u[i])
        xi = (0.5)+((np.sqrt(u[i]/2))*(w0/(width))*(np.sin(theta)))
        eta = ((L0/length)-((w0/length)*(np.sqrt(u[i]/2))*np.cos(theta)))
        integral_xy[i] = np.trapz((sigma_xy*xfunc(xi,ll)*yfunc(eta,mm)), x=theta)
    #plt.plot(u,integral_xy)                 # Plot of first integral vs. u
    #plt.show()
    return (np.trapz(integral_xy,x=u))          # Calculation and return of 2nd integral
#print(double_integral(1,0))

# Function to get gammaEigSquared for respective l, m and n values
def gammaEigSquared(ll,mm,nn):
    return (np.square(a*ll*math.pi) + np.square((mm+0.5)*math.pi) + np.square(b*nn*math.pi))
#print(gammaEigSquared(2,2,2))

# Function of Z_analytical
def z_ana(nn):
    if nn == 0:
        return (1 - np.exp(-2 * kpp * height)) / (2 * kpp * height)
    else:
        return ((np.sqrt(2)*2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*(-1)**nn)

# Function to get A_lmn for respective values of l, m and n
def A_lmn(ll,mm,nn):
    return ((1/gammaEigSquared(ll,mm,nn))*double_integral(ll,mm)*z_ana(nn))
print("A(000) = ", A_lmn(0,0,0))
print("A(100) = ", A_lmn(1,0,0))

triple_sum_vals = np.zeros((numSum,numSum,numSum))
def U(xi, eta, zeta):
    for mm in range(0, numSum):
        for nn in range(0, numSum):
            for ll in range(0, numSum):
                triple_sum_vals[ll,mm,nn] = A_lmn(ll,mm,nn)*xfunc(xi,ll)*yfunc(eta,mm)*zfunc(zeta,nn)
    triple_sum = np.sum(np.sum(np.sum(triple_sum_vals)))
    return ((((triple_sum))*T0)+T0)

#print("Temp difference at the spot: ", U(0.5, L0/length, 0)-T0, " K")
#Loop for the 2D line plot.
plot_rate = 30                                   # Plotting resolution. Only these values will be calculated
temp = np.zeros(plot_rate)                       # Defining 0 temperature vector for filling in the loop below
dist = np.zeros(plot_rate)

for i in range(0,plot_rate):
    temp[i] = (U(0.5,i/plot_rate,0))
    dist[i] = (i/plot_rate)*length
    print(dist[i],temp[i])                               # Printing calculated temperature values.
plt.plot(dist/10**-6, temp)
plt.xlabel('Length ($\mu$m)', fontsize=10)
plt.ylabel('Temperature (K)', fontsize=10)
plt.title('Temperature profile of NR along the center line')
plt_text = "Spot position = %r*L \nLaser Power = %r mW" % (spotPos, laser_power)
plt.text(0, temp[-10], plt_text)
plt.show()

'''

# Loop for 3D data
x_plot_rate = 30
y_plot_rate = 30
temp = np.zeros((x_plot_rate+1, y_plot_rate+1))
x = np.zeros(x_plot_rate+1)
y = np.zeros(y_plot_rate+1)
for xx in range(0, x_plot_rate+1):
    for yy in range(0, y_plot_rate+1):
        temp[xx,yy] = (U(xx/x_plot_rate, yy/y_plot_rate, 0))
        print(xx/x_plot_rate, yy/y_plot_rate, temp[xx, yy])
        x[xx] = (xx/x_plot_rate)*width*10**6
        y[yy] = (yy/y_plot_rate)*length*10**6

z = temp
x, y = np.meshgrid(x,y)
#print(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, z.transpose(), cmap=cm.jet, linewidth=0, antialiased=True)
fig.colorbar(surf, shrink=0.7, aspect=10)
ax.set_xlabel('Width ($\mu$m)', fontsize=10)
ax.set_ylabel('Length ($\mu$m)', fontsize=10)
ax.set_zlabel('Temperature (K)')

plt.show()
'''

