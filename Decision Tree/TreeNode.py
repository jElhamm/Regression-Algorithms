# The `TreeNode` class represents a node in a decision tree used for regression. 
# The node can have various attributes:

# - `feature_index` represents the index of the feature used for splitting at this node.
# - `threshold` represents the threshold value for the feature used for splitting at this node.
# - `left` represents the left child node.
# - `right` represents the right child node.
# - `value` represents the predicted value at this node (leaf node).

# This class is used to construct the decision tree by recursively creating nodes and connecting 
# them based on the splitting criterion. Each node can have child nodes (left and right) or be 
# a leaf node with a predicted value.



class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold     = threshold
        self.left  = left
        self.right = right
        self.value = value