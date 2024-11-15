'''Implentation of the SLAM algorithm on occupancy grid map'''
from ekf_slam import EKF_SLAM

if __name__ == '__main__':
    slam = EKF_SLAM()
    slam.run()