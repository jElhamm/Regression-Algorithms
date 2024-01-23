class RegressionModelFactory:
    _instance = None

    class __RegressionModelFactory:
        def __init__(self):
            pass

        @staticmethod
        def create_model(X, y, degree):
            from PolynomialRegression import PolynomialRegressionModel
            return PolynomialRegressionModel(X, y, degree)

    def __new__(cls):
        if not RegressionModelFactory._instance:
            RegressionModelFactory._instance = RegressionModelFactory.__RegressionModelFactory()
        return RegressionModelFactory._instance
    


# Singleton Design Pattern:
# The code implements the Singleton design pattern. 
# The RegressionModelFactory class is designed to have only one instance.
# The __new__ method overrides the default behavior of Python's object creation mechanism,
# ensuring that only one instance of the __RegressionModelFactory inner class is created and returned.
# The __RegressionModelFactory inner class is where the actual model creation takes place,
# specifically, the creation of a PolynomialRegressionModel object using the create_model method.
# The outer RegressionModelFactory class simply provides access to the single instance of the inner class.
# This design pattern can be useful when there should be exactly one instance of a class throughout the application.