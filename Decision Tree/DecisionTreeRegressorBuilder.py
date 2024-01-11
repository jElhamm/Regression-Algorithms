# The DecisionTreeRegressorBuilder class is a builder class for constructing instances 
# of the DecisionTreeRegressor class.
# It allows you to set the necessary parameters for building a decision tree regressor.



from DecisionTreeRegressor import DecisionTreeRegressor


class DecisionTreeRegressorBuilder:
    def __init__(self):
        self.X = None
        self.y = None
        self.max_depth = None

    def set_X(self, X):
        self.X = X
        return self

    def set_y(self, y):
        self.y = y
        return self

    def set_max_depth(self, max_depth):
        self.max_depth = max_depth
        return self

    def build(self):
        return DecisionTreeRegressor(self.X, self.y, self.max_depth)