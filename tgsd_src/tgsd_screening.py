from typing import Tuple
import numpy as np

class TGSD_Screening:
    def __init__(self, x: np.ndarray, left: np.ndarray, right: np.ndarray):
        self.data = x
        self.right, self.left = right, left
        self.lam_max, self.Bn = None, None
        self.is_unique, self.js_unique = None, None

    def find_lam_max(self) -> np.ndarray:
        BijF = np.dot(np.linalg.norm(self.left).T, np.linalg.norm(self.right.T))
        x = self.data.flatten()
        Xn = self.data / np.linalg.norm(x)
        B = np.dot(self.left.T, np.dot(Xn, self.right.T))
        self.Bn = np.abs(B / BijF)
        self.lam_max = np.max(self.Bn)
        return self.lam_max

    def ST2_2D(self, lam: int) -> Tuple[np.ndarray, np.ndarray]:
        tmp = self.lam_max*(1-2*np.sqrt(1/np.square(self.lam_max) - 1)*(self.lam_max/lam - 1))
        selected = self.Bn < tmp

        sis, sjs = np.where(selected == 0)
        self.is_unique = np.array(np.unique(sis))
        self.js_unique = np.array(np.unique(sjs))
        return self.is_unique, self.js_unique

    def make_dictonary (self):
        if self.left.size != 0:
            for row in range(self.left.shape[0]):
                if row not in self.is_unique:
                    for col in range(self.left.shape[1]):
                        self.left[row][col] = 0


        if self.right.size != 0:
            for row in range(self.right.shape[0]):
                if row not in self.js_unique:
                    for col in range(self.right.shape[1]):
                        self.right[row][col] = 0

        return self.left, self.right



#X = np.random.randn(5, 10)
#D = np.random.randn(5, 8)
#R = np.random.randn(15, 10)

#tgsd = TGSD_Screening(X, D, R)
#tgsd.find_lam_max()
#[is_unique, js_unique] = tgsd.ST2_2D(0.8 * tgsd.lam_max)

#print(is_unique)
#print(js_unique)
