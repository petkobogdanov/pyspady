import unittest
import pandas as pd
import numpy as np


class TestYConversion(unittest.TestCase):
    def test_y_complex_conversion(self, i, Y):
        filename = f'y_iterations_demo_mat/iteration_{i}.csv'
        df = pd.read_csv(filename, header=None)
        df = df.replace({'i': 'j'}, regex=True)
        # Convert complex number strings to complex numbers
        y_prime = df.map(np.complex128).to_numpy()
        cosine_similarity = np.dot(y_prime.flatten(), Y.flatten()) / (np.linalg.norm(y_prime.flatten()) * np.linalg.norm(Y.flatten()))
        print(f'i: {i}, cos y', cosine_similarity)
        #y_prime_minus_y = np.linalg.norm(y_prime - Y, ord=1)
        #print(f"i {i}, norm y_prime minus y = {y_prime_minus_y}")  # This value should be less than 1
        #self.assertTrue(y_minus_y_prime < 1)
