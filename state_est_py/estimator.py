"""Estimation using Dead Reckoning, KF, and EKF for 16-362: Mobile Robot Algorithms Laboratory
"""

import numpy as np


class Estimator:
    """Estimator object containing various methods to estimate state of a differential
    drive robot.

    Attributes:
        r: (float) Radius of the wheels of the robot.
        L: (float) Distance between the two wheels.
        dt: (float) Timestep size.
    """

    def __init__(self, dt=0.04, radius=0.033, axle_length=0.16):
        self.r = radius
        self.L = axle_length
        self.dt = dt

    def dead_reckoning(self, x, u):
        """Estimate the next state using the previous state x and the control
        input u via Dead Reckoning.

        Args:
            x: (np.array for 3 floats) State at the previous timestep
            u: (np.array for 2 floats) Control input to the left (u[0]) and right (u[1]) wheels

        Returns:
            next_x: (np.array of 3 floats) State at the current timestep
        """
        # TODO: Assignment 2, Problem 2.1

        mixer = np.array(
            [[self.r * np.cos(x[2]) / 2, self.r * np.cos(x[2]) / 2],
             [self.r * np.sin(x[2]) / 2, self.r * np.sin(x[2]) / 2],
             [-self.r / self.L         , self.r / self.L          ]]
        )
        x_dot = mixer @ u

        next_x = x + x_dot * self.dt
        return next_x

    def kalman_filter(self, x, P, u, y, Q, R):
        """Localization via Kalman Filtering.

        Estimate the next state and the associated uncertainty from the previous state x
        and uncertainty P, control input u, observation y, process noise covariance Q, and
        measurement noise covariance R. Note that in this case we are ignoring heading state (ψ).

        Args:
            x: (np.array for 2 floats) State at the previous timestep
            P: (np.array of floats, 2x2) Uncertainty in the state at the previous timestep
            u: (np.array for 2 floats) Control input to the left (u[0]) and right (u[1]) wheels
            y: (np.array for 2 floats) GPS observation of the position (x, y)
            Q: (np.array of floats, 2x2) Process model noise covariance matrix
            R: (np.array of floats, 2x2) Measurement model noise covariance matrix

        Returns:
            (next_x, next_P): Tuple(np.array of 2 floats, np.array of floats with shape 2x2)
                              Next state vector and covariance matrix
        """
        # TODO: Assignment 2, Problem 2.2
        #we assume angle is pi/4
        psi = np.pi / 4

        A = np.array([[1, 0], 
                      [0, 1]])
        B = np.array([[self.dt * self.r * np.cos(psi) / 2, self.dt * self.r * np.cos(psi) / 2],
                      [self.dt * self.r * np.sin(psi) / 2, self.dt * self.r * np.sin(psi) / 2]])
        C = np.array([[1, 0], 
                      [0, 1]])
        
        x_t = A @ x + B @ u #curr_state_est
        P_t = A @ P @ A.T + Q #curr state covariance
        K_t = P_t @ C.T @ np.linalg.inv(C @ P_t @ C.T + R) #Kalman gain

        x_t = x_t + K_t @ (y - C @ x_t)
        P_t = (np.eye(2) - (K_t @ C)) @ P_t

        return x_t, P_t

    def extended_kalman_filter(self, x, P, u, y, Q, R):
        """Localization via Extended Kalman Filtering.

        Estimate the next state and the associated uncertainty from the previous state x
        and uncertainty P, control input u, observation y, process noise covariance Q, and
        measurement noise covariance R. Note that in this case we are not ignoring heading state (ψ).

        Args:
            x: (np.array for 3 floats) State at the previous timestep
            P: (np.array of floats, 3x3) Uncertainty in the state at the previous timestep
            u: (np.array for 2 floats) Control input to the left (u[0]) and right (u[1]) wheels
            y: (np.array for 2 floats) GPS observation of the position (x, y)
            Q: (np.array of floats, 3x3) Process model noise covariance matrix
            R: (np.array of floats, 2x2) Measurement model noise covariance matrix

        Returns:
            (next_x, next_P): Tuple(np.array of 3 floats, np.array of floats with shape 3x3)
                              Next state vector and covariance matrix
        """
        # TODO: Assignment 2, Problem 2.3

        dgdx = np.array([[1, 0, (-self.dt * self.r * np.sin(x[2]) * (u[0] + u[1])) / 2],
                         [0, 1, (self.dt * self.r * np.cos(x[2]) * (u[0] + u[1])) / 2],
                         [0, 0, 1]])

        A = dgdx
        B = np.array([[self.dt * self.r * np.cos(x[2]) / 2, self.dt * self.r * np.cos(x[2]) / 2],
                      [self.dt * self.r * np.sin(x[2]) / 2, self.dt * self.r * np.sin(x[2]) / 2]])
        C = np.array([[1, 0, 0], 
                      [0, 1, 0]])
        
        x_t = self.dead_reckoning(x, u) #curr_state_est
        P_t = A @ P @ A.T + Q #curr state covariance
        K_t = P_t @ C.T @ np.linalg.inv(C @ P_t @ C.T + R) #Kalman gain

        x_t = x_t + K_t @ (y - C @ x_t)
        P_t = (np.eye(3) - (K_t @ C)) @ P_t

        return x_t, P_t

        raise NotImplementedError
