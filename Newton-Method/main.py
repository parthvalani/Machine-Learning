import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from Model import LogisticRegressionUsingGD 
# Load the dataset 
def load_data(path, header): 
    marks_df = pd.read_csv(path, header=header) 
    return marks_df 
if __name__ == "__main__": 
    # load the data from the file 
    data = load_data("data/myfile.txt", None)  
    # X = feature values 
    X = data.iloc[:, :-1]  
    # y = target values 
    y = data.iloc[:, -1]  
    # filter out the applicants that got admitted 
    admit = data.loc[y == 1] 
 
    # filter out the applicants that din't get admission 
    not_admit = data.loc[y == 0]  
    # plots 
    plt.scatter(admit.iloc[:, 0], admit.iloc[:, 1], s=10, label='Admit') 
    plt.scatter(not_admit.iloc[:, 0], not_admit.iloc[:, 1], s=10, label='Not Admit')  
    # preparing the data for building the model  
    X = np.c_[np.ones((X.shape[0], 1)), X] 
    y = y[:, np.newaxis] 
    theta = np.zeros((X.shape[1], 1))  
    # Logistic Regression using Gradient Descent 
    model = LogisticRegressionUsingGD() 
    model.fit(X, y, theta) 
    accuracy = model.accuracy(X, y.flatten()) 
    parameters = model.w_ 
    print("The accuracy: {}".format(accuracy))  
    x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)] 
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2] 
    # Plot the decision boundary 
    plt.plot(x_values, y_values, label='Decision Boundary') 
 
    # Set the label 
    plt.xlabel('Marks in 1st Sem') 
    plt.ylabel('Marks in 2nd Sem') 
    plt.legend() 
    plt.show()