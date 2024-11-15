'''Implentation of the EKF-SLAM (unknown location)'''

from typing import List, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt

class EKF_SLAM:

    def __init__(self) -> None:
        #TODO: add inits

        # EKF state covariance
        self.Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)])**2 # Change in covariance

        #  Simulation parameter
        self.Qsim = np.diag([0.2, np.deg2rad(1.0)])**2  # Sensor Noise
        self.Rsim = np.diag([1.0, np.deg2rad(10.0)])**2 # Process Noise

        self.DT = 0.1  # time tick [s]
        self.SIM_TIME = 50.0  # simulation time [s]
        self.MAX_RANGE = 20.0  # maximum observation range
        self.M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
        self.STATE_SIZE = 3  # State size [x,y,yaw]
        self.LM_SIZE = 2  # LM state size [x,y]

        self.show_animation = True

    def ekf_slam(self, X, P, u, z) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Performs 1 iteration of iterative EKF-SLAM through the prediction and update step

        Args:
            X: the state belief of the previous step
            P: the uncertainty of the previous step
            u: the control applied to the agent
            z: the measurement applied

        Returns:
            X_next: the state belief of the next step
            P_next: the associated covariance
        '''
        # Predict
        X, P = self.predict(X, P, u)
        initP = np.eye(2)

        # Update
        X, P = self.update(X, P, z, initP)

        return X, P


    def predict(self, X, P, u) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Performs the predict step of EKF-SLAM

        Args:
            X: the state belief of the previous step
            P: the uncertainty of the previous step
            u: the control applied to the agent

        Returns:
            X_next: the state belief of the next step
            P_next: the associated covariance
        '''


        #TODO: change name
        G, Fx = self.jacob_motion(X, u)
        #Fx is identity matrix of the corresponding size

        X[:self.STATE_SIZE] = self.apply_control(X[:self.STATE_SIZE], u)
        P = G.T @ P @ G + Fx.T @ self.Cx @ Fx
        return X, P
    

    def apply_control(self, X, u):
        '''
        Gets the motion model through current state and input

        Args:
            X: state estimation
            u: control input

        Returns:
            X: state after control is applied
        '''
        F = np.eye(3)

        B = np.array([[self.DT * math.cos(X[2, 0]), 0],
                      [self.DT * math.sin(X[2, 0]), 0],
                      [0.0, self.DT]])
        
        X = (F @ X) + (B @ u)
        return X
    

    def update(self, xEst, PEst, z, initP):
        """
        Performs the update step of EKF SLAM

        :param xEst:  nx1 the predicted pose of the system and the pose of the landmarks
        :param PEst:  nxn the predicted covariance
        :param z:     the measurements read at new position
        :param initP: 2x2 an identity matrix acting as the initial covariance
        :returns:     the updated state and covariance for the system
        """
        for iz in range(len(z[:, 0])):  # for each observation
            minid = self.search_correspond_LM_ID(xEst, PEst, z[iz, 0:2]) # associate to a known landmark

            nLM = self.calc_n_LM(xEst) # number of landmarks we currently know about

            if minid == nLM: # Landmark is a NEW landmark
                print("New LM")
                # Extend state and covariance matrix
                xAug = np.vstack((xEst, self.calc_LM_Pos(xEst, z[iz, :])))
                PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), self.LM_SIZE)))),
                                np.hstack((np.zeros((self.LM_SIZE, len(xEst))), initP))))
                xEst = xAug
                PEst = PAug

            lm = self.get_LM_Pos_from_state(xEst, minid)
            y, S, H = self.calc_innovation(lm, xEst, PEst, z[iz, 0:2], minid)

            K = (PEst @ H.T) @ np.linalg.inv(S) # Calculate Kalman Gain
            xEst = xEst + (K @ y)
            PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

        xEst[2] = self.pi_2_pi(xEst[2])
        return xEst, PEst
    
    def calc_innovation(self, lm, xEst, PEst, z, LMid):
        """
        Calculates the innovation based on expected position and landmark position

        :param lm:   landmark position
        :param xEst: estimated position/state
        :param PEst: estimated covariance
        :param z:    read measurements
        :param LMid: landmark id
        :returns:    returns the innovation y, and the jacobian H, and S, used to calculate the Kalman Gain
        """
        delta = lm - xEst[0:2]
        q = (delta.T @ delta)[0, 0]
        zangle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
        zp = np.array([[math.sqrt(q), self.pi_2_pi(zangle)]])
        # zp is the expected measurement based on xEst and the expected landmark position

        y = (z - zp).T # y = innovation
        y[1] = self.pi_2_pi(y[1])

        H = self.jacobH(q, delta, xEst, LMid + 1)
        S = H @ PEst @ H.T + self.Cx[0:2, 0:2]

        return y, S, H

    def jacobH(self, q, delta, x, i):
        """
        Calculates the jacobian of the measurement function

        :param q:     the range from the system pose to the landmark
        :param delta: the difference between a landmark position and the estimated system position
        :param x:     the state, including the estimated system position
        :param i:     landmark id + 1
        :returns:     the jacobian H
        """
        sq = math.sqrt(q)
        G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                    [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

        G = G / q
        nLM = self.calc_n_LM(x)
        F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
        F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                        np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

        F = np.vstack((F1, F2))

        H = G @ F

        return H
    

    def observation(self, xTrue, xd, u, RFID):
        """
        :param xTrue: the true pose of the system
        :param xd:    the current noisy estimate of the system
        :param u:     the current control input
        :param RFID:  the true position of the landmarks

        :returns:     Computes the true position, observations, dead reckoning (noisy) position,
                    and noisy control function
        """
        xTrue = self.apply_control(xTrue, u)

        # add noise to gps x-y
        z = np.zeros((0, 3))

        for i in range(len(RFID[:, 0])): # Test all beacons, only add the ones we can see (within MAX_RANGE)

            dx = RFID[i, 0] - xTrue[0, 0]
            dy = RFID[i, 1] - xTrue[1, 0]
            d = math.sqrt(dx**2 + dy**2)
            angle = self.pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
            if d <= self.MAX_RANGE:
                dn = d + np.random.randn() * self.Qsim[0, 0]  # add noise
                anglen = angle + np.random.randn() * self.Qsim[1, 1]  # add noise
                zi = np.array([dn, anglen, i])
                z = np.vstack((z, zi))

        # add noise to input
        ud = np.array([[
            u[0, 0] + np.random.randn() * self.Rsim[0, 0],
            u[1, 0] + np.random.randn() * self.Rsim[1, 1]]]).T

        xd = self.apply_control(xd, ud)
        return xTrue, z, xd, ud
    
    def calc_n_LM(self, x):
        """
        Calculates the number of landmarks currently tracked in the state
        :param x: the state
        :returns: the number of landmarks n
        """
        n = int((len(x) - self.STATE_SIZE) / self.LM_SIZE)
        return n


    def jacob_motion(self, x, u):
        """
        Calculates the jacobian of motion model.

        :param x: The state, including the estimated position of the system
        :param u: The control function
        :returns: G:  Jacobian
                Fx: STATE_SIZE x (STATE_SIZE + 2 * num_landmarks) matrix where the left side is an identity matrix
        """

        # [eye(3) [0 x y; 0 x y; 0 x y]]
        Fx = np.hstack((np.eye(self.STATE_SIZE), np.zeros(
            (self.STATE_SIZE, self.LM_SIZE * self.calc_n_LM(x)))))

        jF = np.array([[0.0, 0.0, -self.DT * np.multiply(u[0,0], math.sin(x[2, 0]))],
                   [0.0, 0.0, self.DT * u[0,0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]],dtype=float)
        # print(jF)
        #         
        # print(Fx.T @ jF @ Fx)
        # print(Fx.shape)
        # G = np.eye(self.STATE_SIZE) + Fx.T @ jF @ Fx
        G = Fx.T @ jF @ Fx
        if self.calc_n_LM(x) > 0:
            print(Fx.shape)
        return G, Fx


    def calc_LM_Pos(self, x, z):
        """
        Calculates the pose in the world coordinate frame of a landmark at the given measurement.

        :param x: [x; y; theta]
        :param z: [range; bearing]
        :returns: [x; y] for given measurement
        """
        zp = np.zeros((2, 1))

        zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
        zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])
        #zp[0, 0] = x[0, 0] + z[0, 0] * math.cos(x[2, 0] + z[0, 1])
        #zp[1, 0] = x[1, 0] + z[0, 0] * math.sin(x[2, 0] + z[0, 1])

        return zp


    def get_LM_Pos_from_state(self, x, ind):
        """
        Returns the position of a given landmark

        :param x:   The state containing all landmark positions
        :param ind: landmark id
        :returns:   The position of the landmark
        """
        lm = x[self.STATE_SIZE + self.LM_SIZE * ind: self.STATE_SIZE + self.LM_SIZE * (ind + 1), :]

        return lm


    def search_correspond_LM_ID(self, xAug, PAug, zi):
        """
        Landmark association with Mahalanobis distance.

        If this landmark is at least M_DIST_TH units away from all known landmarks,
        it is a NEW landmark.

        :param xAug: The estimated state
        :param PAug: The estimated covariance
        :param zi:   the read measurements of specific landmark
        :returns:    landmark id
        """

        nLM = self.calc_n_LM(xAug)

        mdist = []

        for i in range(nLM):
            lm = self.get_LM_Pos_from_state(xAug, i)
            y, S, H = self.calc_innovation(lm, xAug, PAug, zi, i)
            mdist.append(y.T @ np.linalg.inv(S) @ y)

        mdist.append(self.M_DIST_TH)  # new landmark

        minid = mdist.index(min(mdist))

        return minid

    def calc_input(self):
        v = 1.0  # [m/s]
        yawrate = 0.1  # [rad/s]
        u = np.array([[v, yawrate]]).T
        return u

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    


    def run(self):

        print(" start!!")

        time = 0.0

        # RFID positions [x, y]
        RFID = np.array([[10.0, -2.0],
                        [15.0, 10.0],
                        [3.0, 15.0],
                        [-5.0, 20.0]])

        # State Vector [x y yaw v]'
        xEst = np.zeros((self.STATE_SIZE, 1))
        xTrue = np.zeros((self.STATE_SIZE, 1))
        PEst = np.eye(self.STATE_SIZE)

        xDR = np.zeros((self.STATE_SIZE, 1))  # Dead reckoning

        # history
        hxEst = xEst
        hxTrue = xTrue
        hxDR = xTrue

        while self.SIM_TIME >= time:
            time += self.DT
            u = self.calc_input()

            xTrue, z, xDR, ud = self.observation(xTrue, xDR, u, RFID)

            xEst, PEst = self.ekf_slam(xEst, PEst, ud, z)

            x_state = xEst[0:self.STATE_SIZE]

            # store data history
            hxEst = np.hstack((hxEst, x_state))
            hxDR = np.hstack((hxDR, xDR))
            hxTrue = np.hstack((hxTrue, xTrue))

            if self.show_animation:  # pragma: no cover
                plt.cla()

                plt.plot(RFID[:, 0], RFID[:, 1], "*k")
                plt.plot(xEst[0], xEst[1], ".r")

                # plot landmark
                for i in range(self.calc_n_LM(xEst)):
                    plt.plot(xEst[self.STATE_SIZE + i * 2],
                            xEst[self.STATE_SIZE + i * 2 + 1], "xg")

                plt.plot(hxTrue[0, :],
                        hxTrue[1, :], "-b")
                plt.plot(hxDR[0, :],
                        hxDR[1, :], "-k")
                plt.plot(hxEst[0, :],
                        hxEst[1, :], "-r")
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.001)