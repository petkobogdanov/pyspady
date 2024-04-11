from typing import Tuple
import numpy as np

class TGSD_Screening:
    def __init__(self, x: np.ndarray, right: np.ndarray, left: np.ndarray):
        self.data = x
        self.right, self.left = right, left
        self.lam_max, self.Bn = None, None

    def find_lam_max(self) -> np.ndarray:
        BijF = np.dot(np.linalg.norm(self.left).T, np.linalg.norm(self.right.T))
        x = self.data.flatten()
        Xn = self.data / np.linalg.norm(x)
        B = np.dot(self.left.T, np.dot(Xn, self.right.T))
        self.Bn = np.abs(B / BijF)
        self.lam_max = np.max(self.Bn)  
        return self.lam_max

    def ST2_2D(self, lam: int) -> Tuple[np.ndarray, np.ndarray]:
        tmp = self.lam_max*(1-2*np.sqrt(1/self.lam_max**2 - 1)*(self.lam_max/lam - 1))
        selected = self.Bn < tmp
        
        sis, sjs = np.where(selected == 0)
        is_unique = np.array(np.unique(sis))
        js_unique = np.array(np.unique(sjs))
        return is_unique, js_unique

#X = np.random.randn(5, 10)
#D = np.random.randn(5, 8)
#R = np.random.randn(15, 10)

#tgsd = TGSD_Screening(X, R, D)
#tgsd.find_lam_max()
#[is_unique, js_unique] = tgsd.ST2_2D(0.8 * tgsd.lam_max)

#print(is_unique)
#print(js_unique)