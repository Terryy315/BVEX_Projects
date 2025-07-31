import csv
import matplotlib.pyplot as plt
from datetime import datetime
from geopy.distance import geodesic

# --- Convert DMS to Decimal Degrees ---
def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        decimal *= -1
    return decimal

# --- Load CSV, Compute Accurate Geodesic Distances ---
def compute_geodesic_distances(csv_file, ref_lat_dms, ref_lon_dms):
    ref_lat = dms_to_decimal(*ref_lat_dms)
    ref_lon = dms_to_decimal(*ref_lon_dms)
    ref_point = (ref_lat, ref_lon)

    distances = []
    times = []

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            time_str = row['gps_time']
            time_obj = datetime.strptime(time_str, "%H:%M:%S")  # Modify format if needed

            dist = geodesic((lat, lon), ref_point).meters
            distances.append(dist)
            times.append(time_obj)

    return times, distances

# --- Plot Results ---
def plot_distances_with_time(times, distances):
    plt.figure(figsize=(10, 5))
    plt.scatter(times, distances, marker='o')
    plt.title('Distance to Reference Point')
    plt.xlabel('Time')
    plt.ylabel('Distance (meters)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Replace with your real DMS reference coordinate
    ref_lat_dms = (44, 20, 45.488602, 'N')
    ref_lon_dms = (76, 10, 14.350144, 'W')

    csv_path = '/home/terry/ground_test/gps_data/gps_log_2025-07-25_14_16_13.csv'  # Replace with your actual CSV file path

    times, distances = compute_geodesic_distances(csv_path, ref_lat_dms, ref_lon_dms)
    plot_distances_with_time(times, distances)
