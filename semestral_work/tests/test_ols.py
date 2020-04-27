from OLS.OLS import * 
import numpy as np
import pytest

class TestOLS:

    def test_mse_function(self):
        model = OLS()
        # ((8 - 0) ^ 2 + (8 - 0) ^ 2) / 2
        score = model.score(np.array([0, 0]), np.array([8, 8]))
        assert pytest.approx(score, 0.1) == 64.0
        # ((10 - 8) ^ 2 + (10 - 8) ^ 2) / 2
        score = model.score(np.array([10, 10]), np.array([8, 8]))
        assert pytest.approx(score, 0.1) == 4.0
        # ((1 - 1) ^ 2 + (1 - 1) ^ 2) / 2
        score = model.score(np.array([1, 1]), np.array([1, 1]))
        assert pytest.approx(score, 0.1) == 0

        target = np.random.randint(100)
        prediction = np.random.randint(100)
        score = model.score(target, prediction)
        assert pytest.approx(score, 0.00001) == np.square(prediction - target).mean()

    def test_lin_reg_function(self):
        # TODO
        return

    def test_gd_function(self):
        # TODO
        return

    def test_sgd_function(self):
        #TODO
        return

    def test_mgd_function(self):
        # TODO
        return
