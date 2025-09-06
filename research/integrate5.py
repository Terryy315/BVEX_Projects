import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord,AltAz,EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS
import glob as gb
import os
import datetime as dt
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
import time
from matplotlib.animation import FuncAnimation
import boolean
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt
import sympy
import math
from scipy.ndimage import uniform_filter1d



def ma(data, window_size):
    weights = np.ones(window_size) / window_size
    ma = np.convolve(data, weights, mode='valid')
    return ma


#some variable specific to each dataset/scan
time_limiter_L = 1755452609 #1750039697 #1754445893 #1754445148 #1750039670 #1754017080
time_limiter_H = 1755453521 #1750039949 #1754446200 #1754446500 #1750040303 #1754017440



#load spectrum data, change to appropriate path
datapath = '/home/terry/ground_test/2025-08-17_17-42_BVEX_data'
#/home/terry/ground_test/2025-06-16_01-39_BVEX_data
#/home/terry/ground_test/power_data/2025-08-01_02-25_BVEX_data
#/home/terry/ground_test/power_data/2025-08-06_01-18_BVEX_data
powerpaths = gb.glob(os.path.join(datapath,'*_spectrum_data.txt'))
powerpaths.sort()

allfiles = os.listdir(datapath)
print("All files in datapath:", allfiles)


#load motor elevation path, change to appropriate path
elev_data = np.loadtxt("/home/terry/ground_test/motor_pv_1755450134.txt", delimiter=';')
#/home/terry/ground_test/motor/motor_pv_1750038061.txt
#/home/terry/ground_test/motor_2/motor_pv_1754014952.txt
#/home/terry/ground_test/motor_3/motor_pv_1754443165.txt


fig, ((ax8, ax9),(ax10, ax11)) = plt.subplots(2,2)


#fill elevation arrays
elevTime = []
elevDataReal = []
elevVel = []
j = 0
for x in elev_data[:,1]:
    if (x > 10 and x < 40): #use this to manually limit the elevation data range
        elevTime.append(elev_data[j,0])
        elevDataReal.append(elev_data[j,1])
        elevVel.append(elev_data[j,2])
    j = j +1

elevTime = np.array(elevTime)
elevDataReal = np.array(elevDataReal)
elevVel = np.array(elevVel)

print("Printing elevation data:")
print(elevDataReal)

#load rfsoc arrays
rfsoc_time = []
power = []

#define the frequency range
freqs = np.linspace(21.014e9,21.1098e9,num=150)

for p in powerpaths:
    data = np.loadtxt(p)
    for d in data:
        #use the commented if statement if you need to manually specify a time range
        if (d[0] > time_limiter_L and d[0] < time_limiter_H):
        #if (1):
            #integrate power using trapezoid function 
            rfsoc_time.append(d[0])
            ip=trapezoid(d[51:201],x=freqs)
            power.append(ip)
rfsoc_time = np.array(rfsoc_time)

#power = savgol_filter(power, window_length=7, polyorder=3)
'''
print("Printing power data")
print(power)
'''
#find the time where both rfsoc and elevation data overlap in WRT time
start = rfsoc_time[0]
end = rfsoc_time[-1]
mask = (elevTime >= start) & (elevTime <= end)
elevTime_overlap = elevTime[mask]
elevDataReal_overlap = elevDataReal[mask]
elevVel_overlap = elevVel[mask]

print("rfsoc start, end:")
print(start, end)

print("Printing elevation overlap data")
print(elevDataReal_overlap)

ax8.scatter(rfsoc_time, power, label="Power", color="green")
ax8.set_ylabel("Power")
ax8.set_xlabel("Time")
ax8.set_title("Power/Elevation V Time")
ax8.legend()

ax99 = ax8.twinx()
ax99.plot(elevTime_overlap, elevDataReal_overlap, label="Elevation", color="red")
ax99.set_ylabel("Elevation (deg)")
ax99.legend()

ax10.scatter(elevTime_overlap, elevDataReal_overlap, label="Elevation", color="green")
ax10.set_ylabel("Elevation (deg)")
ax10.set_xlabel("Time")
ax10.set_title("Elevation V Time")
ax10.legend()



#need to interpolate elevation data points to ensure that the plotting function works
fused_elev = np.interp(rfsoc_time, elevTime_overlap, elevDataReal_overlap)
fused_velo = np.interp(rfsoc_time, elevTime_overlap, elevVel_overlap)

'''
#ex2 interpolate power instead
fused_power = np.interp(elevTime_overlap, rfsoc_time, power)
power = fused_power
fused_elev = elevDataReal_overlap
fused_velo = elevVel_overlap
'''

#separate data based on odd/even scan
oddpower = []
oddelev = []
evenpower = []
evenelev = []
oddtime = []
eventime = []

i = 0
for k in fused_velo:
    if (k > 0):
        oddpower.append(power[i])
        oddelev.append(fused_elev[i])
        oddtime.append(elevTime_overlap[i])
    else:
        evenpower.append(power[i])
        evenelev.append(fused_elev[i])
        eventime.append(elevTime_overlap[i])
    i = i +1




#sort all arrays based on elevation, prepare for plotting and fitting, WRT elevation
oddelev = np.array(oddelev)
evenelev = np.array(evenelev)
oddpower = np.array(oddpower)
evenpower = np.array(evenpower)
sorted_idx = np.argsort(oddelev)
oddelev = oddelev[sorted_idx]
oddpower = oddpower[sorted_idx]
sorted_idx = np.argsort(evenelev)
evenelev = evenelev[sorted_idx]
evenpower = evenpower[sorted_idx]

'''
#moving average filter
window = 33  # must be odd
evenpower = uniform_filter1d(evenpower, size=window, mode='nearest')

window = 33  # must be odd
oddpower = uniform_filter1d(oddpower, size=window, mode='nearest')
'''

# Butterworth low-pass filter (keeps array shape)
b, a = butter(N=4, Wn=0.005)  # 4th order, cutoff freq=0.05 (normalized)
evenpower = filtfilt(b, a, evenpower)

b, a = butter(N=4, Wn=0.02)
oddpower = filtfilt(b,a, oddpower)



#define the Gaussian Function
#a1 = baseline amp; a2 = peak; d = offset; peak u = centre; s = peak width
def func(x, a1, a2, d, u, s):
   # return a1 * 1/(np.cos(np.pi / 2 - x*np.pi / 180)) + d
    u = u*np.pi/180
    s = s*np.pi/180
    return a1 * 1/(np.cos(np.pi / 2 - x*np.pi / 180 - 0.7/180 * np.pi)) + a2 * np.exp(-((x*np.pi / 180 - u)**2) / (2*s**2)) + d

fwhm_deg = 0
fwhm_err = 0


#Fit the odd scan data
#popt, pcov = curve_fit(func, oddelev, oddpower, p0=[0.9e17, 1.2e17, 0.5e16, 26, 2])
popt, pcov = curve_fit(func, oddelev, oddpower, p0=[9.35e16, 9.7e16, 0.5e16, 23.5, 1])

perr = np.sqrt(np.diag(pcov))
for name, val, err in zip(['Baseline', 'Peak', 'Offset', 'Centre', 'Sigma (Peak Width)'], popt, perr):
    print(f"{name} = {val:.3e} ± {err:.3e}")

print("Result of Covariance Matrix:\n", pcov)

fwhm_deg = 2 * np.sqrt(2 * np.log(2)) * popt[4]
fwhm_err = 2 * np.sqrt(2 * np.log(2)) * perr[4]
print(f"ODD Scan FWHM: {fwhm_deg} ± {fwhm_err}")

#plot the odd scan data
ax9.plot(oddelev, func(np.array(oddelev), *popt), label="Fitted", color="red")
ax9.scatter(oddelev, oddpower, label="Raw", color="green")
ax9.set_ylabel("Odd Power")
ax9.set_xlabel("Elevation")
ax9.set_title(f"Odd (UP) Scan, FWHM: {fwhm_deg} ± {fwhm_err}")
ax9.legend()

#print("ODD Power Best Guess:\n", popt)




#Fit the even scan data
popt, pcov = curve_fit(func, evenelev, evenpower, p0=[1e15, 2e16, 0.9e17, 23.5, 1])

print("Even Power Best Guess:\n", popt)
perr = np.sqrt(np.diag(pcov))
for name, val, err in zip(['Baseline', 'Peak', 'Offset', 'Centre', 'Sigma (Peak Width)'], popt, perr):
    print(f"{name} = {val:.3e} ± {err:.3e}")

print("Result of Covariance Matrix:\n", pcov)

fwhm_deg = 2 * np.sqrt(2 * np.log(2)) * popt[4]
fwhm_err = 2 * np.sqrt(2 * np.log(2)) * perr[4]
print(f"EVEN Scan FWHM: {fwhm_deg} ± {fwhm_err}")


#Plot the even scan data
ax11.plot(evenelev, func(np.array(evenelev), *popt),label="Fitted", color="red")
ax11.scatter(evenelev, evenpower,label="Raw", color="green")
ax11.set_ylabel("Even Power")
ax11.set_xlabel("Elevation")
ax11.set_title(f"Even (Down) Scan, FWHM: {fwhm_deg} ± {fwhm_err}")
ax11.legend()

'''
print("Even Power Best Guess:\n", popt)
perr = np.sqrt(np.diag(pcov))
for name, val, err in zip(['a1', 'a2', 'd', 'u', 's'], popt, perr):
    print(f"{name} = {val:.3e} ± {err:.3e}")

print("Result of Covariance Matrix:\n", pcov)

fwhm_deg = 2 * np.sqrt(2 * np.log(2)) * popt[4]
fwhm_err = 2 * np.sqrt(2 * np.log(2)) * perr[4]
print(f"EVEN Scan FWHM: {fwhm_deg} ± {fwhm_err}")
'''
fig.suptitle("DITHER SCAN, TIME: 1755452609, August 17, 2025 17:43:29 UTC", fontsize=16)


plt.show()




