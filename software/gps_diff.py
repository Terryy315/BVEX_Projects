import csv
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

# --- Convert DMS to Decimal Degrees ---
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

# --- Haversine Distance ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# --- Load CSV and Calculate Distances ---
def compute_distances(csv_file, ref_lat_dms, ref_lon_dms):
    ref_lat = dms_to_decimal(*ref_lat_dms)
    ref_lon = dms_to_decimal(*ref_lon_dms)
    distances = []

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            dist = haversine(lat, lon, ref_lat, ref_lon) * 1000
            distances.append(dist)

    return distances

# --- Plot Results ---
def plot_distances(distances, time):
    plt.figure(figsize=(10, 5))
    plt.scatter(time, distances, marker='o')
    plt.title('Distance to Reference Point')
    plt.xlabel('Time')
    plt.ylabel('Difference (m)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # e.g., reference: 44°13'52" N, 76°30'36" W
    ref_lat_dms = (44, 20, 45.488602, 'N')
    ref_lon_dms = (76, 10, 14.350144, 'W')

    csv_path = '/home/terry/ground_test/gps_data/gps_log_2025-07-25_14_16_13.csv'  # Replace with your file
    distances = compute_distances(csv_path, ref_lat_dms, ref_lon_dms)

    times = []

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            time_str = row['gps_time']
            time_obj = datetime.strptime(time_str, "%H:%M:%S")
            times.append(time_obj)
    plot_distances(distances, times)

