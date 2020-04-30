import numpy as np


class OLSInputError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message


class OLS:
    """
    Class, which implements different methods for calculating least square regression logics
    
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

    def __init__(
        self,
        iterations=10,
        tolerance=0.001,
        learning_rate=0.001,
        batch_size=10,
        type="linear",
    ):
        self.iterations = iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.type = type
        self.slopes = []
        self.slopes_records = []
        self.cost_records = []

    def validate_input_types(func):
        """
        Decorator to control, that the input is the numpy array with the float type
        """

        def function_wrapper(model, *args, **kwargs):
            def arguments_wrapper(*args):
                for arg in args:
                    if not isinstance(arg, np.ndarray):
                        raise OLSInputError(
                            "Input should be of np.array type, got {}".format(type(arg))
                        )

            arguments_wrapper(*args)
            arguments_wrapper(*kwargs.values())

            return func(model, *args, **kwargs)

        return function_wrapper

    def validate_input_shapes(func):
        def parameter_wrapper(model, *args, **kwargs):
            def arguments_wrapper(*args):
                for arg in args:
                    if len(arg.shape) < 2 or arg.shape[1] < 1:
                        raise OLSInputError(
                            "Input should be of np.array with shape (m, k), got {}".format(
                                arg.shape
                            )
                        )

            arguments_wrapper(*args)
            arguments_wrapper(*kwargs.values())

            return func(model, *args, **kwargs)

        return parameter_wrapper

    def validate_nan_samples(func):
        def parameter_wrapper(model, *args, **kwargs):
            def arguments_wrapper(*args):
                for arg in args:
                    if np.isnan(np.min(arg)):
                        raise OLSInputError("Input shouldn't contain nan values")

            arguments_wrapper(*args)
            arguments_wrapper(*kwargs.values())

            return func(model, *args, **kwargs)

        return parameter_wrapper

    @validate_input_types
    @validate_input_shapes
    @validate_nan_samples
    def fit(self, features: np.array, target: np.array):
        """
        Prepare data for the fit:
        
        Add column on the zero index with all ones to the features. 
           The reason is to get rid of bias,
           as each bias can be converted to the feature.
           For example,

           y = 
           b + slope1 * f1 + slope2 * f2 ... = 
           slope0 * 1 + slope1 * f1 + slope2 * f2...
           
        Input:
            features - preprocessed np.array of the shape (m, n)
            target - preprocessed np.array of the shape (m, n)
        """
        if features.shape[0] != target.shape[0]:
            raise OLSInputError(
                "Target and features should have the same shape, got {}, {}".format(
                    features.shape[0], target.shape[0]
                )
            )

        # Add zero column to the features
        features_copy = np.c_[np.ones(features.shape[0]), features.copy()]

        if self.type == "linear":
            self.__linear_regression(features_copy, target)
        elif self.type == "GD" or self.type == "SGD" or self.type == "MGD":
            self.__gradient_descent_fit(features_copy, target)
        else:
            assert "Incorrect type set"

    @validate_input_types
    @validate_input_shapes
    @validate_nan_samples
    def predict(self, features: np.array):
        """
        Predict value for the fitted model
        """
        if len(self.slopes) == 0:
            raise OLSInputError("Fit model before prediction")

        if features.shape[1] != self.slopes.shape[0] - 1:
            raise OLSInputError(
                "Incorrect features columns shape, should be ({}, k)".format(
                    self.slopes.shape[0] - 1
                )
            )

        biased_features = np.c_[np.ones(features.shape[0]), features]

        return self.__predict(biased_features)

    def score(self, target: np.array, prediction: np.array):
        """
        Calculate MSE cost on the predictions. 
        
        Math equasion:
            MSE = (1/n) * sum((prediction - target)^2)
        """
        return np.square(prediction - target).mean()

    def __predict(self, features: np.array):
        """
            Performs the prediction on the data, but doesn't add bias column to the data
        """

        return np.dot(features, self.slopes)

    def __mse_cost(self, features: np.array, target: np.array):
        """
        Calculates mean square error between feature and target.

        Input:
            slopes: -  np.array, vector of caclulated slopes
            features: - np.array, a row of the feature matrix
            target: - np.array, the target
        """

        prediction = self.__predict(features)

        return self.score(target, prediction)

    def __did_reach_tolerance(self, offset=2):
        """
        Calculates, if the gradient descent has reached the best predicted value.
        
        Input:
            delta - number, which shows to which record the last record is compared to, 
                    e. g. if the offset is n, then take element at the n index from the end of
                    the history and compare with the last item
        """
        if len(self.cost_records) >= offset:
            return (
                abs(self.cost_records[-offset] - self.cost_records[-1]) < self.tolerance
            )

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

        self.slopes = np.ones(shape=(features.shape[1], 1))

        self.slopes_records = []

        self.cost_records = []

        descent_function = {
            "GD": self.__gradient_descent,
            "SGD": self.__stochastic_gradient_descent,
            "MGD": self.__minibatch_gradient_descent,
        }

        descent_function[self.type](features, target)

    def __minibatch_gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The minibatch gradient descent takes a batch of the n examples and adjust slopes during iteration
        """

        for iteration in range(self.iterations):
            index = np.random.randint(0, features.shape[0], size=self.batch_size)

            self.__gradient_descent_iteration(features[index, :], target[index])

            if self.__did_reach_tolerance():
                return

    def __stochastic_gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The stochastic gradient descent takes only one random example during the iteration and adjust slopes
        """
        for iteration in range(self.iterations):
            index = np.random.randint(0, features.shape[0])

            feature_item = features[index, :].reshape(1, features.shape[1])
            target_item = target[index].reshape(1, 1)

            self.__gradient_descent_iteration(feature_item, target_item)
            # Delta is set to the 10, cuz of the random unstable cases. For example, the same
            # element is taken for 3 or more times
            if self.__did_reach_tolerance(offset=5):
                return

        return

    def __gradient_descent(self, features: np.array, target: np.array):
        """
        Perform gradient descent for n iterations.
        
        The basic gradient descent takes all data and calculates the result
        """
        for iteration in range(self.iterations):
            self.__gradient_descent_iteration(features, target)

            if self.__did_reach_tolerance():
                return

    def __gradient_descent_iteration(
        self, features: np.array, target: np.array, records=True
    ):
        """
        Calculates gradient descent for the features and target.

        Input:
            slopes: - np.array, vector of slopes (share: (k, 1), 
              where k is number of feature columns)
            features: - np.array, matrix of features (shape: (n, m))
            targets: - np.array, matrix of targets (shape: (n, 1))
            records: - bool, shows, if should record data

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

        predictions = self.__predict(features)

        gradient_slopes = 2 * (
            features.T.dot(predictions - target).mean(axis=1)
        ).reshape(self.slopes.shape)

        self.slopes -= self.learning_rate * gradient_slopes

        if records:
            self.slopes_records += [self.slopes.copy()]

        if records:
            self.cost_records += [self.__mse_cost(features, target)]
