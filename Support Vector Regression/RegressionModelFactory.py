# This is a Python factory class named 'RegressionModelFactory' that provides a static 
# method 'create_model' to instantiatea custom Support Vector Regression (SVR) model.



from CustomSVR import CustomSVR

class RegressionModelFactory:
    @staticmethod
    def create_model(X, y, kernel='rbf', C=1.0, epsilon=0.1):
        return CustomSVR(X, y, kernel, C, epsilon)