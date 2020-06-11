import matplotlib.pyplot as plt  
# Ploting the scatter graph 
def scatter_plot(x, y, size=10, x_label='x', y_label='y', color='b'): 
    plt.scatter(x, y, s=size, color=color) 
    set_labels(x_label, y_label)  
# Set the labels for axis 
def set_labels(x_label, y_label): 
    plt.xlabel(x_label) 
    plt.ylabel(y_label) 
    plt.show()