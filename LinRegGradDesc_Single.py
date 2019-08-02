#Step 0: Include the required libraries for data access
import numpy as np
from array import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


#Step 1. Retrieve the data from the file
df = pd.read_csv('fuel_consumption.csv')

#Step 2. Extract the column 'x' values from csv file and store in variable x
x = df['ENGINESIZE']

#Step 3. Extract the column 'y' values from csv file and store in variable x
y = df['CO2EMISSIONS']

#Step 4. Create  1-d array of ones
ones = np.ones(len(x))

#Step 5. Update variable 'x' as a 2-D array where each row is of the form [1 x(0);1 x(1), ...]
#IMPORTANT: Dimension of this variable matrix 'x' would be m x 2, where m is length of no. of inputs
x = np.transpose(np.array([ones , x]))

# Step 6. Store the no. of inputs in variable m; this would be equal to length of 'y' column vector defined in step 3
m = len(y)

# Step 7. Initialize a variable array called 'theta' that will hold the parameters theta0 and theta1.
# theta0 is the bias and theta1 is the slope, in the equation that is being modelled:
# h = theta0 + theta1 * x
theta = np.array([0.00,0.00])

# Step 8. Initialize a variable array called 'h' which is going to be a column vector with length equals to that of variable y (original output)
h = np.zeros(len(y))

#Step 9. Define the learning rate, alpha
alpha = 0.02

#Step 10. Define the no. of Iterations
noOfIter = 2000

# Step 11. Initializing variable for using in while loop to calculate the costs for every pair of theta
i=0

# Step 12. Initialize a  column vector 'arrCosts' that will hold the values of calculated cost for every pair of theta.
# The size of this array would be equal to no. of iteration. Every iteration will generate a cost value
arrCosts = np.zeros(noOfIter)

# Step 13. Initialize a  column vector 'arrCosts' that will hold the values of every pair of theta.
# The size of this array would be equal to no. of iteration as #rows and 2 columns (each of theta0 and theta1).
arrThetas = np.zeros((noOfIter , 2))

# Step 14. Implementing gradient descent algorithm
while i<noOfIter:  #in every iteration

    # (i) store the theta/parameter values for future comparison in arrThetas
    arrThetas[i][0] = theta[0]
    arrThetas[i][1] = theta[1]

    # (ii) calculate the prediction using existing theta
    # note: dim(x)=100x2, dim(h)=100 x 1, hence to calculate 'h',
    # array 'x' should be multipled with matrix of dim(2x1),
    # and that can be achieved by multiplying 'x' with transpose of vector 'theta'
    h = np.dot(x , np.transpose(theta))
    
    # (iii) calculate the cost, i.e., J for given pair of parameters theta[0], theta[1],
    # 'h' which is the predicted output and 'y'
    cost = sum(np.square(h-y))*(1/(2*m))

    # (iv) save the value of 'cost' variable in the array arrCosts for future comparison
    # ith index of arrCosts will contain cost value generated in i-th iteration
    arrCosts[i] = cost

    # (v) update values of theta[0] and theta[1] for next iteration
    theta[0]=theta[0]-((alpha/m)*sum(h-y))
    theta[1]=theta[1]-(alpha/m)*sum((h-y)*x[:,1])

    # (vi) update the iterator i for continuing the loop
    i=i+1

#Step 15: Save the value of lowest cost in variable 'lowestCost'
lowestCost=min(arrCosts)

#Step 16: Save the value of index in which value of lowest cost occurs
temp=np.where(arrCosts==lowestCost)
position=temp[0][0]

print(f"Lowest cost = {lowestCost}")

# Step 17: Save the values of the intercept (or theta[0]) and coefficient( or theta[1]) in separate variables
intercept = round(arrThetas[position][0],2)
coefficient = round(arrThetas[position][1],2)
print(f"Optimum Parameters:")
print(f"Intercept: {intercept}")
print(f"Coeffcient: {coefficient}")

print(f"Linear Regression Model: {intercept} + ({coefficient})*x")

#Step 18(i): Test the fitment of the model built
y_prediction=np.dot(x , np.transpose(arrThetas[position]))

x_feature=x[:,1]

#Step 18(ii): Plot the real output 'y' and predicted output 'y_prediction' against input colummn vector 'x_feature'
plt.plot(x_feature,y,'x',color="red")
plt.plot(x_feature,y_prediction,color="blue")
plt.show()

plt.plot(range(noOfIter),arrCosts,'red')




















