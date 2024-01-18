import unittest
import pandas as pd
import numpy as np


class TestWConversion(unittest.TestCase):
    def test_w_complex_conversion(self, i, W):
        filename = f'psi_phi_orth_w/w_iteration_{i}.csv'
        df = pd.read_csv(filename, header=None)
        df = df.replace({'i': 'j'}, regex=True)
        # Convert complex number strings to complex numbers
        w_prime = df.map(np.complex128).to_numpy()
        cosine_similarity = np.dot(w_prime.flatten(), W.flatten()) / (np.linalg.norm(w_prime.flatten(), ord=2) * np.linalg.norm(W.flatten(), ord=2))
        print(f'i: {i}, cos w', cosine_similarity)
        return cosine_similarity
        #self.assertTrue(w_prime_minus_w < 1)
