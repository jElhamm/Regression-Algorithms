# This code defines a RegressionModelFactory class, which is responsible for creating instances of the RidgeRegressionModel.
# The factory follows the Singleton design pattern, ensuring that only one instance of the RidgeRegressionModel is created.



from RidgeRegressionModel import RidgeRegressionModel


class RegressionModelFactory:
    _instance = None

    @staticmethod
    def create_model(X, y, alpha):
        if RegressionModelFactory._instance is None:
            # Create an instance of RidgeRegressionModel if it does not already exist
            RegressionModelFactory._instance = RidgeRegressionModel(X, y, alpha)
        return RegressionModelFactory._instance