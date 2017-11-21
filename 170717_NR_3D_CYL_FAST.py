'''
This is a code for calculating the Temperature gradient on the surface of a cantilever, irradiated with a Gaussian
laser spot at a specified position on the surface centerline.
Based on theory by James Davis - Refer to Fiber Heating 8.0.docx

Author: Anupum Pant
Date: June 13, 2017
Updated: July 17, 2017
'''

import numpy as np
import math
import matplotlib.pyplot as plt
#from scipy.integrate import simps
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import os

# Set variables here
length = 20e-6                      # Length of the NR (m)
width = 5e-6                        # Width of the NR (m)
height = 110e-9                      # Height of the NR (m)
T0 = 77                             # Cryostat temperature (K)
Tinf = 298                          # Ambient temperature (K)
kappa = 20                          # Thermal conductivity (W/(m.K))
n_CdS = 2.5513                      # Refractive index real part (532 nm)
k_CdS = 0.0057657                   # Refractive index imag part (532 nm)
#n_CdS = 2.6399                     # Refractive index real part (514 nm)
#k_CdS = 0.055754                   # Refractive index imag part (514 nm)
ref = 0.19081                       # Reflectivity (532 nm)
#ref = 0.20317                      # Reflectivity (514 nm)
spotDiameter = 2.5e-6               # Laser spot diameter (m)
w0 = spotDiameter/2                 # Laser spot radius (m)
sampleRate = 15                     # Spatial sample rate
numSum = 5                          # Eigenvalue sample rate
lambda0 = 532e-9                    # laser wavelength (m)
laser_power = 0.035                     # Laser power in milliwatts
Power = laser_power*10**-3*(1-ref)  # Laser power (W)
spotPos = 0.75                      # Normalized position of laser spot (0: base, 1: tip)
base_area = width*height

# Few calculations, using defined variables above.
r_max = np.sqrt(0.5*np.log(100)*np.square(w0))
r_max = width/2
kpp = (2*math.pi*k_CdS)/(lambda0)
L0 = spotPos*length
I0 = (Power*2)/(math.pi*np.square(w0))
f = 1-np.exp(-2*kpp*height)
Force = (((f*(1-ref)+2*ref)*math.pi*w0**2*I0)/(2*300000000))*(1-np.exp(-width**2/(2*w0**2)))
print(Force)

theta = np.linspace(0, 2*math.pi, sampleRate)
r = np.linspace(0, r_max, sampleRate)
u_max = (2*np.square(r_max))/(np.square(w0))
#u = (2*np.square(r))/(np.square(w0))
u = np.linspace(0, u_max, sampleRate)
a = length/width
b = length/height
l = np.linspace(0, numSum, numSum+1)
m = np.linspace(0, numSum, numSum+1)
n = np.linspace(0, numSum, numSum+1)
Q_gen = 0.5*I0*math.pi*np.square(w0)*(1-np.exp(-2*kpp*height))*(1-np.exp(-u_max))

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
dbl_int = np.zeros((numSum,numSum,numSum))
z_ana = np.zeros((numSum,numSum,numSum))
gammaEigSquared = np.zeros((numSum,numSum,numSum))
for ll in range(0, numSum):
    for mm in range(0, numSum):
        for nn in range(0, numSum):
            integral_xy = np.zeros(sampleRate)
            for i in range(0,sampleRate):
                sigma_xy = ((0.5*w0**2*length*kpp)/(width*kappa*T0))*I0*np.exp(-u[i])
                xi = (0.5)+((np.sqrt(u[i]/2))*(w0/(width))*(np.sin(theta)))
                eta = ((L0/length)-((w0/length)*(np.sqrt(u[i]/2))*np.cos(theta)))
                integral_xy[i] = np.trapz((sigma_xy*xfunc(xi,ll)*yfunc(eta,mm)), x=theta)
            dbl_int[ll,mm,nn] = (np.trapz(integral_xy,x=u))          # Calculation and return of 2nd integral
            gammaEigSquared[ll,mm,nn] = (np.square(a*ll*math.pi) + np.square((mm+0.5)*math.pi) + np.square(b*nn*math.pi))
            if nn == 0:
                z_ana[ll,mm,nn] = (1 - np.exp(-2 * kpp * height)) / (2 * kpp * height)
            else:
                z_ana[ll,mm,nn] = ((np.sqrt(2)*2*kpp*height)/(np.square(2*kpp*height)+np.square(nn*math.pi)))*(1-np.exp(-2*kpp*height)*(-1)**nn)
            A_lmn = ((1 / gammaEigSquared) * dbl_int * z_ana)
            #print("Status", ((ll/(numSum-1))*100)"% complete")
print(dbl_int)
print("A(000) = ", A_lmn[0,0,0])
print("A(100) = ", A_lmn[1,0,0])
print("A(010) = ", A_lmn[0,1,0])
print("A(001) = ", A_lmn[0,0,1])
print(gammaEigSquared[0,0,0])

# Function to get A_lmn for respective values of l, m and n
triple_sum_vals = np.zeros((numSum+1,numSum+1,numSum+1))
def U(xi, eta, zeta):
    for ll in range(0, numSum):
        for mm in range(0, numSum):
            for nn in range(0, numSum):
                triple_sum_vals[ll,mm,nn] = A_lmn[ll,mm,nn]*xEigFun(xi,ll)*yEigFun(eta,mm)*zEigFun(zeta,nn)
    triple_sum = np.sum(np.sum(np.sum(triple_sum_vals)))
    return ((((triple_sum))*T0))
#print(U(0.5,0.75,0))

#print("Temp difference at the spot: ", U(0.5, L0/length, 0)-T0, " K")
#Loop for the 2D line plot.
plot_rate = 50                                   # Plotting resolution. Only these values will be calculated
temp = np.zeros(plot_rate+1)                       # Defining 0 temperature vector for filling in the loop below
dist = np.zeros(plot_rate+1)

file_name = "Full_report.txt"
f = open(file_name, 'a' )
line1 = 'PARAMETERS: Length = %r um || ' % (length*10**6)
line2 = 'Width = %r um|| ' % (width*10**6)
line3 = 'Height = %r nm|| ' % (height*10**9)
line4 = 'Laser Power = %r mW || ' % laser_power
line5 = 'Spot Position = %r*length || ' % spotPos
line6 = 'Eigen values = %r || ' % numSum
line7 = 'Spatial subdivisions = %r || ' % sampleRate
line8 = 'Plot resolution = %r || ' % plot_rate
line9 = 'Refractive index = %r + %ri || ' % (n_CdS, k_CdS)
f.write(line1 + line2 + line3 + line4 + line5 + line6 + line7 + line8 + line9 + '\n')
'''
#Width plot. at widthplot_position*L
widthplot_position = 0.75
for i in range(0,plot_rate+1):
    temp[i] = (U(i/plot_rate,widthplot_position,0))
    dist[i] = (i/plot_rate)*width
    tempvsdist = "%r    %r\n" % (dist[i],temp[i])
    f.write(tempvsdist)   # Printing calculated temperature values.
f.write('==========================================================\n\n')
plt.plot(dist/10**-6, temp)
plt.xlabel('Length ($\mu$m)', fontsize=10)
plt.ylabel('Temperature (K)', fontsize=10)
plt.title('Temperature profile of NR along the width (2pi integration)')
plt_text = "Spot position = %r*L \nLaser Power = %r mW" % (spotPos, laser_power)
plt.text(length*10**6-7, 77, plt_text)
plt.show()
f.close()

'''
#Length plot along centerline
for i in range(0,plot_rate+1):
    temp[i] = (U(0.5, i/plot_rate,0))
    dist[i] = (i/plot_rate)*length
    tempvsdist = "%r    %r\n" % (dist[i],temp[i])
    f.write(tempvsdist)   # Printing calculated temperature values.
f.write('==========================================================\n\n')
plt.plot(dist/10**-6, temp)
plt.xlabel('Length ($\mu$m)', fontsize=10)
plt.ylabel('$\Delta T_{max}$ (K)', fontsize=10)
plt.title('Temperature profile of NR along the center line')
plt_text = "Spot position = %r*L \nLaser Power = %r mW" % (spotPos, laser_power)
plt.text(length*10**6-7, 77, plt_text)
plt.show()
f.close()


slope = (temp[5]-temp[0])/(dist[5]-dist[0])
print("Base flux = ", kappa*slope*base_area)
print("Q generated = ", Q_gen)


# 3D plot
x_plot_rate = 50
y_plot_rate = 50
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
surf = ax.plot_surface(x, y, z.transpose(), cmap=cm.Spectral_r, linewidth=0, antialiased=True, alpha=1)
fig.colorbar(surf, shrink=0.7, aspect=10)
ax.set_xlabel('Width ($\mu$m)', fontsize=18, labelpad=10)
ax.set_ylabel('Length ($\mu$m)', fontsize=18, labelpad=10)
ax.set_zlabel('$\Delta T_{max}$ (K)', fontsize=18, labelpad=10)
ax.tick_params(labelsize=18)
ax.set_xlim3d(-((length-width)/2)*10**6, (width+(length-width)/2)*10**6)



plt.show()
print("Max T", np.amax(np.amax(z)))
print("Max Del_T", np.amax(np.amax(z))-T0)

