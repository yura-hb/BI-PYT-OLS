from OLS.OLS import * 
import numpy as np
import pytest

class TestOLS:

    def test_mse_function(self):
        """
        Test correct implementation of the cost function
        """
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
        """
        Test lin regression function
        """
        self.fit_constant_function()
        self.fit_identical_function()
        self.fit_identical_noise_function()

    def test_gd_function(self):
        self.fit_identical_noise_function(iterations=100, learning_rate = 0.001, type='GD')
        self.fit_cubic_function(iterations=100, tolerance=0, learning_rate=0.001, type='GD')

    def test_sgd_function(self):
        self.fit_identical_noise_function(iterations=1000, tolerance=0.00001, learning_rate=0.001, type='SGD')

    def test_mgd_function(self):
        self.fit_identical_noise_function(iterations=1000, tolerance=0.00001, learning_rate=0.001, batch_size=20, type='MGD')
    
    @staticmethod
    def fit_constant_function(**kwargs):
        """
        Tests if constant function is fit correctly
        """
        model = OLS(**kwargs)
        model.fit(
            np.array([10, 20, 30, 40], dtype=np.float).reshape(4, 1), 
            np.array([10, 10, 10, 10], dtype=np.float).reshape(4, 1)
        )
        assert pytest.approx(model.slopes[0], 0.0001) == 10
        assert pytest.approx(model.slopes[1], 0.0001) == 0
    
    @staticmethod
    def fit_identical_function(**kwargs):
        model = OLS(**kwargs)
        model.fit(
            np.linspace(0, 1, 100).reshape(100, 1), 
            np.linspace(0, 1, 100).reshape(100, 1)
        )
        
        assert pytest.approx(model.slopes[0], 0.0001) == 0
        assert pytest.approx(model.slopes[1], 0.0001) == 1
        
        
    @staticmethod
    def fit_identical_noise_function(**kwargs):
        """
        Gradient descent method can't correctly find weight with the straight line, as mse error
        is the polynom with the power of 2 and single dot with zero mse. 
        However, if we add some noise to the data, approximation will work better
        """
        
        model = OLS(**kwargs)
        model.fit(
            np.linspace(0, 1, 100).reshape(100, 1) - (np.random.random(100) * 0.2).reshape(100, 1), 
            np.linspace(0, 1, 100).reshape(100, 1)
        )
        
        sample = np.array([1]).reshape(1, 1)
        
        print(model.slopes)
        
        assert pytest.approx(1, 0.2) == model.predict(sample)[0][0]        

    @staticmethod
    def fit_cubic_function(**kwargs):
        """
        Gradient descent tries to minize mse error, for the qubic polynom, minimal error is line y=0
        """
        model = OLS(**kwargs)
        
        model.fit(
            np.linspace(-1, 1, 200).reshape(200, 1), 
            np.poly1d([1, 0, 0, 0])(np.linspace(-1, 1, 200)).reshape(200, 1)
        )
        
        assert pytest.approx(model.slopes[0], 0.0001) == 0
        assert pytest.approx(model.slopes[0], 0.0001) == 0
