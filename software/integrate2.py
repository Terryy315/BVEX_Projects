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
from scipy.signal import find_peaks, peak_widths
import time
from matplotlib.animation import FuncAnimation
import boolean
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit
import sympy
import math
datapath = '/home/terry/ground_test/2025-06-16_01-39_BVEX_data'
powerpaths = gb.glob(os.path.join(datapath,'*_spectrum_data.txt'))
powerpaths.sort()

allfiles = os.listdir(datapath)
print("All files in datapath:", allfiles)

# Load heading data
alt_data = np.loadtxt('/home/terry/Desktop/GPS/time_heading.txt', delimiter=',')
altTime = alt_data[100:, 0]
altData = alt_data[100:, 1]
altData = np.exp(0 - abs(altData - 135.33)) 

# Load elevation Data
elev_data = np.loadtxt("/home/terry/ground_test/motor/motor_pv_1750038061.txt", delimiter=';')
elevTime = elev_data[:, 0]
elevDataReal = elev_data[:, 1]
elevData = np.exp(0 - abs(elevDataReal - 24.75)) 
#elevInterp = np.interp(altTime, elevTime, elevDataReal)


# Load velocity Data
elevVel = elev_data[:, 2]


#put markers on elevation
marker_elev,_ = find_peaks(elevData, height=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)


fig, ((ax1, ax3), (ax4, ax5), (ax6, ax7), (ax8, ax9), (ax10, ax11)) = plt.subplots(5,2) 
  
ax1.set_xlabel('Unix time') 
ax1.set_ylabel('Integrated Power', color = 'red') 

# Adding Twin Axes

ax2 = ax1.twinx() 
  
ax2.set_ylabel('Relative to true position (max=1)', color = 'blue') 
ax2.plot(altTime, altData, label = "Heading", color = 'blue')
ax2.plot(elevTime, elevData, markevery=marker_elev, label = "Elevation", color = 'purple') 
ax2.tick_params(axis ='y', labelcolor = 'blue') 


freqs = np.linspace(21.014e9,21.1098e9,num=150)
rfsoc_time = []
power = []
i=1
for p in powerpaths:
    data = np.loadtxt(p)
    for d in data:
        if (d[0] > 1750039670 and d[0] < 1750040303): 
            rfsoc_time.append(d[0])
            ip=trapezoid(d[51:201],x=freqs)
            power.append(ip)
rfsoc_time = np.array(rfsoc_time)
power = np.array(power)


marker,_ = find_peaks(power, height=1e17, distance=20, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)

marker_times = rfsoc_time[marker]

#Interpolate elevation at those times
corresponding_values2 = np.interp(marker_times, elevTime, elevDataReal)

#Print result
elev_result = []
for idx, t, v2 in zip(marker, marker_times, corresponding_values2):
    print(f"Index in rfsoc power: {idx}, Time: {t:f}, Elevation at this time: {v2:f}")
    if ((v2 < 27.00) and (v2 > 22.00)):
        elev_result.append(v2)

elev_result = np.array(elev_result)
print(elev_result)

# Absolute differences
truevalue = 24.75
diffs = np.abs(elev_result - truevalue)

# Average difference
average_diff = np.mean(diffs)

print("AVG ABS Elevation Difference: ", average_diff)

diffs = elev_result - truevalue

# Average difference
average_diff = np.mean(diffs)

print("AVG Elevation Difference: ", average_diff)


#plot the difference
ax3.hist(diffs,bins=100, color='skyblue', edgecolor='black')

# Adding labels and title
ax3.set_xlabel('Elevation Offset')
ax3.set_ylabel('Count')
ax3.set_title('Elevation Offset Histogram')


#calculate peak width (beam angle)
pw = peak_widths(power, marker, rel_height=0.5, prominence_data=None, wlen=None)
pw_info = pw[0]

pw_info_filtered = []
for p in pw_info:
    if (p < 50):
        pw_info_filtered.append(p)
pw_info_filtered = np.array(pw_info_filtered)

pw_info = np.array(pw_info)

print('Peak width: ', pw_info)

fused_vel = np.interp(marker_times, elevTime, elevVel)

bw_result = []
for idx, t, v2 in zip(marker, marker_times, fused_vel):
    print(f"Index in rfsoc power: {idx}, Time: {t:f}, Velocity at this time: {v2:f}")
    bw_result.append(v2)

for i in marker:
    print(elevVel[i-10])

bw_result = np.array(bw_result) * pw_info

bw_filtered = []
for bw in bw_result:
    if (bw < 10 and bw > -10):
        bw_filtered.append(bw)
bw_filtered = np.array(bw_filtered)
print('Beam Width: ', bw_result)
print("Average filtered BW: ", np.mean(bw_filtered))

#plot the difference
ax4.hist(bw_result,bins=100, color='skyblue', edgecolor='black')

# Adding labels and title
ax4.set_xlabel('Beam Width')
ax4.set_ylabel('Count')
ax4.set_title('Beam Width Histogram')


ax1.plot(rfsoc_time, power,'-gD', markevery=marker, color = 'red')
ax1.tick_params(axis= 'y', labelcolor = 'red')

peak = rfsoc_time[np.argmax(power)]
print(dt.datetime.utcfromtimestamp(peak))
print(peak)

'''
ax5.plot(power)
ax5.plot(marker, power[marker], "x")
ax5.hlines(*pw[1:], color="C2")
'''

#power vs elevation
start = rfsoc_time[0]
end = rfsoc_time[-1]
mask = (elevTime >= start) & (elevTime <= end)
elevTime_overlap = elevTime[mask]
elevDataReal_overlap = elevDataReal[mask]
elevVel_overlap = elevVel[mask]

fused_elev = np.interp(rfsoc_time, elevTime_overlap, elevDataReal_overlap)
fused_velo = np.interp(rfsoc_time, elevTime_overlap, elevVel_overlap)

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
        oddtime.append(rfsoc_time[i])
    else:
        evenpower.append(power[i])
        evenelev.append(fused_elev[i])
        eventime.append(rfsoc_time[i])
    i = i +1

ax1.set_xlabel('Unix time') 
ax1.set_ylabel('Integrated Power', color = 'red') 

ax5.scatter(evenelev, evenpower)
ax5.set_ylabel('Even Power', color = 'blue')
ax5.set_xlabel('Even Elevation', color = 'blue')
ax5.set_title('Even Scan (DOWN)', color = 'blue')

coeffs = np.polyfit(evenelev, evenpower, deg=1)
baseline = np.polyval(coeffs, evenelev)
evenpower_clean = evenpower - baseline

ax6.scatter(evenelev, evenpower_clean)
ax6.set_ylabel('Even Power', color = 'blue')
ax6.set_xlabel('Even Elevation', color = 'blue')
ax6.set_title("Fitted Even Scan", color='blue')

ax7.scatter(oddelev, oddpower)
ax7.set_ylabel('Odd Power', color = 'blue')
ax7.set_xlabel('Odd Elevation', color = 'blue')
ax7.set_title('Odd Scan (UP)', color = 'blue')

coeffs = np.polyfit(oddelev, oddpower, deg=1)
o_baseline = np.polyval(coeffs, oddelev)
oddpower_clean = oddpower - o_baseline


ax8.scatter(oddelev, oddpower_clean)
ax8.set_ylabel('Odd Power', color = 'blue')
ax8.set_xlabel('Odd Elevation', color = 'blue')
ax8.set_title("Fitted Odd Scan", color='blue')


evenpower_clean = np.array(evenpower_clean)
oddpower_clean = np.array(oddpower_clean)
oddtime = np.array(oddtime)
eventime = np.array(eventime)

power_clean = np.concatenate((oddpower_clean, evenpower_clean))
elev_clean = np.concatenate((oddelev, evenelev))
time_clean = np.concatenate((oddtime, eventime))

print(time_clean)

sorted_idx = np.argsort(time_clean)

# Step 2: Apply the same indices to all arrays
time_clean = time_clean[sorted_idx]
power_clean = power_clean[sorted_idx]
elev_clean = elev_clean[sorted_idx]

'''
ax9.scatter(time_clean, power_clean)
ax9.set_ylabel('Clean Power')
ax9.set_xlabel('Clean Time')
ax9.set_title('Clean Power VS Time')
'''
#ax10.plot(time_clean, power_clean)
ax9.set_ylabel('Clean Power')
ax9.set_xlabel('Clean Time')
ax9.set_title('Clean Power VS Time')


#re-compute beam width
marker,_ = find_peaks(power_clean, height=1e16, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
marker_times = time_clean[marker]
print(marker.shape)
ax9.plot(time_clean, power_clean,'-gd',  markevery=marker, color='purple')
ax9.tick_params(axis= 'y', labelcolor = 'purple')

'''
#calculate peak width (beam angle)
pw = peak_widths(power_clean, marker, rel_height=0.5, prominence_data=None, wlen=None)
pw_info = pw[0]

pw_info_filtered = []
for p in pw_info:
    if (p < 50):
        pw_info_filtered.append(p)
pw_info_filtered = np.array(pw_info_filtered)

pw_info = np.array(pw_info)

print('Peak width: ', pw_info)

fused_vel = np.interp(marker_times, elevTime, elevVel)

bw_result = []
for idx, t, v2 in zip(marker, marker_times, fused_vel):
    print(f"Index in clean power: {idx}, Time: {t:f}, Velocity at this time: {v2:f}")
    bw_result.append(v2)


bw_result = np.array(bw_result)
bw_result = bw_result * pw_info

bw_filtered = []
for bw in bw_result:
    if (bw < 10 and bw > -10):
        bw_filtered.append(bw)
bw_filtered = np.array(bw_filtered)
print('Beam Width: ', bw_result)
print("Average filtered BW: ", np.mean(bw_filtered))

#plot the difference
ax11.hist(bw_result,bins=100, color='skyblue', edgecolor='black')

# Adding labels and title
ax11.set_xlabel('Beam Width')
ax11.set_ylabel('Count')
ax11.set_title('Beam Width Histogram')

'''

#sort based on elevation
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

evenelev_copy = []
evenpower_copy = []


for i in range(0, len(evenelev)):
    if not(evenelev[i] > 24.51 and evenelev[i] < 26 and evenpower[i] < 9.27e16):
        evenelev_copy.append(evenelev[i])
        evenpower_copy.append(evenpower[i])

evenelev = np.array(evenelev_copy)
evenpower = np.array(evenpower_copy)

#a1 = baseline amp; a2 = peak; d = offset; peak u = centre; s = peak width
def func(x, a1, a2, d, u, s):
   # return a1 * 1/(np.cos(np.pi / 2 - x*np.pi / 180)) + d
    u = u*np.pi/180
    s = s*np.pi/180
    return a1 * 1/(np.cos(np.pi / 2 - x*np.pi / 180 - 0.7/180 * np.pi)) + a2 * np.exp(-((x*np.pi / 180 - u)**2) / (2*s**2)) + d

popt, pcov = curve_fit(func, oddelev, oddpower, p0=[0.9e17, 1.2e17, 0.5e16, 26, 2])
#elev = np.linspace(23,29,num=1000)

#ax11.scatter(elev, func(np.array(elev), 0.5, 3, 10, 10, 1))

ax10.plot(oddelev, func(np.array(oddelev), *popt), label="Fitted", color="red")
ax10.scatter(oddelev, oddpower, label="Raw", color="green")
ax10.set_ylabel("Odd Power")
ax10.set_xlabel("Elevation")
ax10.set_title("Odd Scan")
ax10.legend()
print("Odd Power Best Guess:\n", popt)
perr = np.sqrt(np.diag(pcov))
for name, val, err in zip(['a1', 'a2', 'd', 'u', 's'], popt, perr):
    print(f"{name} = {val:.3e} ± {err:.3e}")

print("Result of Covariance Matrix:\n", pcov)

popt, pcov = curve_fit(func, evenelev, evenpower, p0=[1e15, 2e16, 0.9e17, 24, 2.1])
for x in range (0,1000):
    #print("Even Power Best Guess:", popt)
    popt, pcov = curve_fit(func, evenelev, evenpower, p0=popt)

ax11.plot(evenelev, func(np.array(evenelev), *popt),label="Fitted", color="red")
ax11.scatter(evenelev, evenpower,label="Raw", color="green")
ax11.set_ylabel("Even Power")
ax11.set_xlabel("Elevation")
ax11.set_title("Even Scan")
ax11.legend()
print("Even Power Best Guess:\n", popt)
perr = np.sqrt(np.diag(pcov))
for name, val, err in zip(['a1', 'a2', 'd', 'u', 's'], popt, perr):
    print(f"{name} = {val:.3e} ± {err:.3e}")

print("Result of Covariance Matrix:\n", pcov)

'''
ax7 = ax6.twinx()
ax7.set_ylabel('Power', color = 'blue') 
ax7.plot(rfsoc_time, fused_power, label = "Power", color = 'red')
ax7.tick_params(axis ='y', labelcolor = 'blue') 
'''


plt.legend()
plt.show()
