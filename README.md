# ECE276: LiDAR-Based SLAM

This project implement a Simultaneous Localization and Mapping (SLAM) framework for a differential-drive robot. Initial odometry, constructed by fusing encoder and IMU data, was refined via LiDAR scan matching using the Iterative Closest Point (ICP) algorithm and further optimized through factor graph optimization using the GTSAM library. We generate 2-D occupancy grids and texture maps to evaluate the effectiveness of these refinement methods. The implementation of fixed-interval and proximity-based loop-closure constraints within the factor graph demonstrates a significant reduction in accumulated error and ensures global consistency in the robot trajectory. The final results show a precise recovery of the environment's geometry by applying these two trajectory refinement algorithms.

## Features
* **Odometry Fusion:** Fusing encoder and IMU data for initial pose estimation.
* **Scan Matching:** 2D ICP implementation using the Kabsch algorithm.
* **Factor Graph Optimization:** Utilizing the **GTSAM** library for global trajectory refinement.
* **Loop Closure:** Fixed-interval and proximity-based constraint detection.
* **Mapping:** 2D occupancy grid generation using Bresenham's ray tracing and RGB-D texture mapping.

## File Structure
```text
ECE276A_PR2/
├── code/
│   ├── icp_2D.py          # 2D ICP implementation using Kabsch algorithm
│   ├── load_data.py       # Load .npz datasets
│   ├── main.py            # Core SLAM algorithm
│   ├── map.py             # Occupancy grid and Bresenham's ray tracing
│   └── icp_warm_up/       
│       ├── data/          # Canonical models
│       ├── icp_3D.py      # 3D ICP implementation
│       └── utils.py       # PC reading and 3D visualization utilities
├── data/                  # Lidar, IMU, and Encoder sensor data
└── dataRGBD/              # RGB and Disparity images for texture maps
```
---

## Requirements
* Python 3.x
* GTSAM
* NumPy
* SciPy
* Matplotlib
* OpenCV (cv2)

---

## Getting start
1 Create and activate a virtual environment
```text
conda create --name venv
conda activate venv
```

2 Install dependencies
```text
conda install numpy scipy matplotlib opencv
pip install gtsam
```

3 Run the program
To execute “SLAM”
```text
cd code
python3 main.py
```

To excite “3D-ICP”
```text
cd icp_warm_up
python3 icp_3D.py
```

## Outputs
The program generates the following visualizations:
* Initial trajectory (IMU + Encoders)
* Occupancy Grid and Texture Maps before optimization
* Refined Occupancy Grid and Texture Maps after Pose Graph Optimization

---
*Developed as part of ECE 276A: Sensing and Estimation Robotics at UC San Diego.*