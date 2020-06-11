import numpy as np 
import matplotlib.pyplot as plt 
from LinearModel import LinearRegressionUsingGD 
from Metrics import Performance 
from Plots import scatter_plot 
import random 
def generate_data_set(): 
    #Generates Random Data 
    # Generate 500 points between -pi/2 and pi/2 
    Theta1 = np.random.rand(500) * np.pi - np.pi / 2 
    # Generate 500 points between pi/2 and 3pi/2 
    Theta2 = np.random.rand(500) * np.pi + np.pi / 2 
    # Radius = 2 
    R = 2 
    # center of upper half moon 
    C1 = [-1, 0] 
    # center of lower half moon 
    C2 = [1, 0]  
    x1 = np.zeros((500, 2)) 
    x1[:, 0] = R * np.sin(Theta1) + C1[0] + .6 * np.random.rand(500) 
    x1[:, 1] = R * np.cos(Theta1) + C1[1] + .6 * np.random.rand(500) 
    x2 = np.zeros((500, 2)) 
    x2[:, 0] = R * np.sin(Theta2) + C2[0] + .6 * np.random.rand(500) 
    x2[:, 1] = R * np.cos(Theta2) + C2[1] + .6 * np.random.rand(500) 
    Y = np.zeros(1000) 
    Y[:500] = 0 
    Y[500:] = 1  
    # array for shuffle 
    X = np.zeros((1000, 3)) 
    X[:, :2] = np.concatenate((x1, x2)) 
    X[:, 2] = Y 
    random.shuffle(X) 
    # Deliver data 
    a = X[:, :2] 
    b = X[:, 2] 
    plt.scatter(a[:, 0], a[:, 1], c=b, cmap='winter') 
    plt.axis([-4, 4, -4, 4]) 
    plt.axis('on') 
    plt.show() 
    return a,b 
if __name__ == "__main__": 
    # initializing the model 
    linear_regression_model = LinearRegressionUsingGD()  
    # generate the data set 
    x, y = generate_data_set()  
    # adding 1 to all the instances of the training set. 
    m = x.shape[0] 
    x_train = np.c_[np.ones((m, 1)), x]  
    # train the model 
    linear_regression_model.fit(x_train, y)  
    # predict the values 
    predicted_values = linear_regression_model.predict(x_train) 
   # Cost_function 
    cost_function = linear_regression_model.cost_ 
   # plotting 
    scatter_plot(x, y)  
    # computing metrics 
    metrics = Performance(y, predicted_values) 
    # compute root mean square error 
    rmse = metrics.compute_rmse()  
    # print the error 
    print('Root mean squared error is {}.'.format(rmse))