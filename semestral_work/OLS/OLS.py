import numpy as np

class OLS:
    """
    Class, which implements different methods for calculating least square logics
    
    Input:

        1. iterations:   - Number of the iterations, default is 10
        2. tolerance:    - The point, when gradient descent stops execution, 
                           default is 0.001
        3. learning_rate:- The speed of the minimization
        4. type:         - The type of the optimization method
        
           There are next options:
           
               1. Linear - calculates the statistic model of the OLS
               2. GD - gradient descent, the basic implementation of the GD with fitting values
               3. SGD - stochastic gradient descent
               4. MGD - minibatch gradient descent
               
               In case of GD, SGD and MGD iterations, tolerance and learning_rate
               affects model performance and slopes_records, cost_records are set
               for better visualisation
    """
    def __init__(self, 
                 iterations = 10, 
                 tolerance = 0.001, 
                 learning_rate = 0.001, 
                 batch_size = 10, 
                 type = 'linear'):
        self.iterations = iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.type = type
        self.slopes = []
        self.slopes_records = []
        self.cost_records = []

    def fit(self, features: np.array, target: np.array):
        """
        Prepare data for the fit:
        
        1. Add column on the zero index with all ones to the features. 
           The reason is to get rid of bias,
           as each bias can be converted to the feature.
           For example,

           y = 
           b + slope1 * f1 + slope2 * f2 ... = 
           slope0 * 1 + slope1 * f1 + slope2 * f2...
        
        """
        # Check of the same type of the value
        assert features.dtype == np.float and target.dtype == np.float
        # Check for NaN
        assert not np.isnan(np.min(features)) and not np.isnan(np.min(target))
        
        # Add zero column to the features
        features_copy = np.ones((features.shape[0], features.shape[1] + 1))
        features_copy[:, 1:] = features
      
        if self.type == 'linear':
            self.__linear_regression(features_copy, target)
        elif self.type == 'GD' or self.type == 'SGD' or self.type == 'MGD':
            self.__gradient_descent_fit(features_copy, target)
        else:
            assert "Incorrect type set"

    def predict(self, features: np.array):
        """
        Predict value for the fitted model
        """
        
        # Check if the shape is valid
        assert features.shape[1] == self.slopes.shape[0] - 1
        
        biased_features = np.c_[np.ones(features.shape[0]), features]
        
        return np.dot(biased_features, self.slopes)
    
    def score(self, target: np.array, prediction: np.array):
        """
        Calculate MSE cost on the predictions. 
        
        Math equasion:
            MSE = (1/n) * sum((prediction - target)^2)
        """
        return np.square(prediction - target).mean() 

    def __mse_cost(self, features: np.array, target: np.array):
        """
        Calculates mean square error between feature and target.

        Input:
            slopes: -  np.array, vector of caclulated slopes
            features: - np.array, a row of the feature matrix
            target: - np.array, the target
        """

        prediction = self.predict(features)

        return score(target, prediction)
    
    def __did_reach_tolerance(self, features: np.array, target: np.array):
        """
        Calculates if gradient descent hits bottom
        """
        if len(self.cost_records) >= 2:
            return abs(self.cost_records[-2] - self.cost_records[-1]) < self.tolerance

        return False
                                
    def __linear_regression(self, features: np.array, target: np.array):
        """
        To find the best slopes use the next formula:
        
        squared_matrix = features.T * features
        inversed_matrix = inverse(squared_matrix)
        result_matrix = features.T * target
        slopes = inversed_matrix * result_matrix
        
        In case if features.T * features is singular matrix,
        throws exception
        
        TODO: - Add validation for the linear independency
        REFERENCE: - https://www.stat.purdue.edu/~boli/stat512/lectures/topic3.pdf
        """
        self.slopes = np.linalg.inv(features.T @ features) @ (features.T @ target)
    
    def __gradient_descent_fit(self, features: np.array, target: np.array):
        """
        Prepare data for the fit:

        0. Validate the correct shapes of the features and targets.
           The shape of the features is (n, k), where the shape of the target
           is (n, 1) and skip rows with missing values.

        1. Setup random slopes vector of shape (k)

        2. Invalidate records
        """

        self.slopes = np.ones(shape=(feature.shape[1], 1))

        self.slopes_records = []

        self.cost_records = []

        descent_function = {
            'GD': self.__gradient_descent,
            'SGD': self.__stochastic_gradient_descent,
            'MGD': self.__minibatch_gradient_descent
        }
        
        descent_function[self.type](features, target)
        
    
    def __minibatch_gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The minibatch gradient descent takes a batch of the n examples and adjust slopes during iteration
        """                   
        
        for iteration in range(self.iterations):
           index = np.random.randint(0, features.shape[0])

           self.__gradient_descent_iteration(features[index, :], target[index])
        
    def __stochastic_gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The stochastic gradient descent takes only one random example during the iteration and adjust slopes
        """
        for iteration in range(self.iterations):
           index = np.random.randint(0, features.shape[0])

           self.__gradient_descent_iteration(features[index, :], target[index])
        return

    def __gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The basic gradient descent takes all data and calculates the result
        """
        for iteration in range(self.iterations):
            self.__gradient_descent_iteration(features, target)

    def __gradient_descent_iteration(self, features: np.array, target: np.array):
        """
        Calculates gradient descent for the features and target.

        Input:
            slopes: - np.array, vector of slopes (share: (k, 1), 
              where k is number of feature columns)
            features: - np.array, matrix of features (shape: (n, m))
            targets: - np.array, matrix of targets (shape: (n, 1))

        Algorithm:
            Variables:
                * iterations: - maximum number of the iterations
                * tolerance: - the stop value of the GD
                * learning_rate - the step size of the function

            Steps:
                1. Slopes are choosen randomly at the iteration 0.
                   Then they are calculated for each feature attribute.

                   The formula of the descent is the next:

                   gradient_slope = (2/n) / sum(prediction - target)
                   It is the derivation of the loss/cost function.

                2. After acquiring gradient_slope update slopes
                   with paying attention to the learning weight.

                   slopes -= learning_rate * gradient_slope
        """
        self.slopes_records += self.slopes

        predictions = self.predict(features)

        gradient_slopes = 2 * (features.T.dot(predictions - target).mean())

        self.slopes -= self.learning_rate * gradient_slopes
        
        self.cost_records += [self.__mse_cost(features, target)]
