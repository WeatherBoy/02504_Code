# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Trying to do exercise 2.10 as a script because it seemed like I had troubles do it in an interactive
# kernel.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# Created by Felix Bo Caspersen, s183319 on Wed Feb 15 2023

#%% IMPORTS ***************************************************************************************
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


#%% HELPER FUNCTIONS ******************************************************************************
def load_im(path: str) -> np.ndarray:

    im = cv2.imread(path)[:, :, ::-1]
    im = im.astype(np.float64) / 255

    return im


def normalize2d(P: np.ndarray) -> np.ndarray:
    """
    Creates the T matrix from a given matrix!
    """
    mu_x = np.mean(P[0, :])
    mu_y = np.mean(P[1, :])
    sigma_x = np.std(P[0, :])
    sigma_y = np.std(P[1, :])

    T_inv = np.array([[sigma_x, 0, mu_x], [0, sigma_y, mu_y], [0, 0, 1]])

    T = np.linalg.inv(T_inv)

    return T


def hest_v2(Q1: np.ndarray, Q2: np.ndarray, normalize=True) -> np.ndarray:
    """
    Takes two points in 2D and returns the estimated homography matrix.
    """

    if len(Q1) != len(Q2):
        raise ValueError("There must be an equal amount of points in the two sets!")

    if normalize:
        T1 = normalize2d(Q1)
        T2 = normalize2d(Q2)
        Q1 = T1 @ Q1
        Q2 = T2 @ Q2

    Bi = []
    for i in range(Q1.shape[1]):
        qi = Q1[:, i]  # <-- getting the first column

        # Creating that weird qx matrix for the Kronecker product
        q1x = np.array([[0, -1, qi[1]], [1, 0, -qi[0]], [-qi[1], qi[0], 0]])

        q2t_hom = Q2[:, i].reshape(-1, 1)  # <-- getting the first column and reshaping does for dim: (1, ) -> (1,1)
        Bi.append(np.kron(q2t_hom.T, q1x).reshape(3, 9))  # <-- formula follows that of week 2, slide 56

    B = np.concatenate(Bi, axis=0)

    # Some TA prooved that it was unneseccary to find their dot product
    # BtB = B.T @ B
    V, Lambda, Vt = np.linalg.svd(B)
    Ht = Vt[-1, :]

    Ht = np.reshape(Ht, (3, 3))
    H = Ht.T

    if normalize:
        H = np.linalg.inv(T1) @ H @ T2

    return H


#%% EX 2.10 ***************************************************************************************

patrick_1 = load_im(path="./Ex02/ims/patrick_01.jpg")
patrick_2 = load_im(path="./Ex02/ims/patrick_02.jpg")
POINTS_PATH_1 = "./Ex02/data/patrick_1_pnts.npy"
POINTS_PATH_2 = "./Ex02/data/patrick_2_pnts.npy"
FIGSIZE = (16, 9)
NUM_POINTS = 4

if not os.path.exists(POINTS_PATH_1):
    plt.figure(figsize=FIGSIZE)
    plt.imshow(patrick_1)
    plt.axis("off")
    patrick_1_pnts = np.asarray(plt.ginput(NUM_POINTS))
    np.save(POINTS_PATH_1, patrick_1_pnts)
else:
    patrick_1_pnts = np.load(POINTS_PATH_1)

if not os.path.exists(POINTS_PATH_2):
    plt.figure(figsize=FIGSIZE)
    plt.imshow(patrick_2)
    plt.axis("off")
    patrick_2_pnts = np.asarray(plt.ginput(NUM_POINTS))
    np.save(POINTS_PATH_2, patrick_2_pnts)
else:
    patrick_2_pnts = np.load(POINTS_PATH_2)

print(patrick_1_pnts)
print(patrick_2_pnts)

# >>>>>>>>>> <<<<<<<<<<
# Getting a test point!
# >>>>>>>>>> <<<<<<<<<<

TEST_POINT_PATH = "./Ex02/data/test_point.npy"

if not os.path.exists(TEST_POINT_PATH):
    plt.figure(figsize=FIGSIZE)
    plt.imshow(patrick_1)
    plt.axis("off")
    test_point = np.asarray(plt.ginput(1))
    np.save(TEST_POINT_PATH, test_point)
else:
    patrick_1_pnts = np.load(TEST_POINT_PATH)

print(test_point)
# %%
