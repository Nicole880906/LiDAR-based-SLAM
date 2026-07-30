import numpy as np
from pathlib import Path


def load_data(data_dir="../data_sim/env1"):
  """Load encoder / LiDAR / IMU logs from an env folder."""
  data_dir = Path(data_dir)

  with np.load(data_dir / "Encoders.npz") as data:
    encoder = {
      "counts" : data["counts"], # 4 x n encoder counts
      "stamps" : data["time_stamps"] # encoder time stamps
    }

  with np.load(data_dir / "Hokuyo.npz") as data:
    lidar = {
      "angle_min" : data["angle_min"], # start angle of the scan [rad]
      "angle_max" : data["angle_max"], # end angle of the scan [rad]
      "angle_increment" : data["angle_increment"], # angular distance between measurements [rad]
      "range_min" : data["range_min"], # minimum range value [m]
      "range_max" : data["range_max"], # maximum range value [m]
      "ranges" : data["ranges"],       # range data [m] (Note: values < range_min or > range_max should be discarded)
      "stamps" : data["time_stamps"]  # acquisition times of the lidar scans
    }
    
  with np.load(data_dir / "Imu.npz") as data:
    imu = {
      "angular_velocity" : data["angular_velocity"], # angular velocity in rad/sec
      "linear_acceleration" : data["linear_acceleration"], # accelerations in gs (gravity acceleration scaling)
      "stamps" : data["time_stamps"]  # acquisition times of the imu measurements
    }
  
  return encoder, lidar, imu


if __name__ == '__main__':
    encoder, lidar, imu = load_data("../data_sim/env1")
    print("Data loaded successfully!")
