import numpy as np   
class Performance: 
    # Defines methods to evaluate the model  
    def __init__(self, y_actual, y_predicted): 
        self.y_actual = y_actual 
        self.y_predicted = y_predicted  
    # fun for calculate error 
    def compute_rmse(self): 
        # Compute the root mean squared error  
        return np.sqrt(self.sum_of_square_of_residuals())  
    def sum_of_square_of_residuals(self): 
        return np.sum((self.y_actual - self.y_predicted) ** 2) 