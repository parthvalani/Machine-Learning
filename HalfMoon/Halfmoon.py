import numpy as np 
import matplotlib.pyplot as plt 
import random  
# Generate 500 points between -pi/2 and pi/2 
T1 = np.random.rand(500) * np.pi - np.pi /2  
# Generate 500 points between pi/2 and 3pi/2 
T2 = np.random.rand(500) * np.pi + np.pi /2  
r = 9 
D1 = [-5, 0] 
D2 = [5, 0]  
X1 = np.zeros((500, 2)) 
X1[:, 0] = r * np.sin(T1) + D1[0] + 3 * np.random.rand(500) 
X1[:, 1] = r * np.cos(T1) + D1[1] + 3 * np.random.rand(500) 
X2 = np.zeros((500, 2)) 
X2[:, 0] = r * np.sin(T2) + D2[0] + 3 * np.random.rand(500) 
X2[:, 1] = r * np.cos(T2) + D2[1] + 3 * np.random.rand(500)  
Y = np.zeros(1000) 
Y[:500] = 0 
Y[500:] = 1  
# array for shuffle 
S = np.zeros((1000, 3)) 
S[:, :2] = np.concatenate((X1, X2)) 
S[:, 2] = Y 
random.shuffle(S)  
# Deliver data 
X = S[:, :2] 
Y = S[:, 2] 
def perceptron_sgd(X, Y): 
    w = np.zeros(len(X[0])) 
    eta = 1 
    epochs = 20  
    for t in range(epochs): 
        for i, x in enumerate(X): 
            if (np.dot(X[i], w) * Y[i]) <= 0: 
                w = w + eta * X[i] * Y[i]  
    return w   
w = perceptron_sgd(X, Y)  
# Generate test data  
# Generate 500 points between -pi/2 and pi/2 
T1 = np.random.rand(500) * np.pi - np.pi / 2 
# Generate 500 points between pi/2 and 3pi/2 
T2 = np.random.rand(500) * np.pi + np.pi / 2  
R = 10 
R1 = [-5, 0] 
R2 = [5, 0]  
X1 = np.zeros((500, 2)) 
X1[:, 0] = R * np.sin(T1) + R1[0] + 3 * np.random.rand(500) 
X1[:, 1] = R * np.cos(T1) + R1[1] + 3 * np.random.rand(500) 
X2 = np.zeros((500, 2))  
X2[:, 0] = R * np.sin(T2) + R2[0] + 3 * np.random.rand(500) 
X2[:, 1] = R * np.cos(T2) + R2[1] + 3 * np.random.rand(500) 
Y = np.zeros(1000) 
Y[:500] = 0 
Y[500:] = 1 
X = np.concatenate((X1, X2)) 
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='spring') 
plt.axis([-20, 20, -20, 20]) 
plt.axis('on') 
plt.show()  
Success = 0 
Error = 0  
for i in X[:500, :]: 
    if np.dot(i, w) <= 0: 
        Success += 1 
    else: 
        Error += 1  
for i in X[500:, :]: 
    if np.dot(i, w) > 0: 
        Success += 1 
    else: 
        Error += 1 
print("Accuracy : " + str(Success / (Success + Error))) 
print("Error : " + str(Error / (Success + Error)))