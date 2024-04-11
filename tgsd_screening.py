import numpy as np


class TGSD_Screening:
    def __init__(self, x -> np.ndarray[any], right, left):
        self.data = x
        self.right = right
        self.left = left
        self.lam_max = None

    def lam_max(self):
        BijF = np.linalg.norm(self.left, axis=1)[:, np.newaxis].T @ np.linalg.norm(self.right, axis=1)[:, np.newaxis]
        x = self.data.flatten()
        xN = self.data / np.linalg.norm(x)
        B = np.transpose(self.left) @ xN @ self.right
        Bn = np.abs(B / BijF)
        self.lam_max = np.max(Bn)




