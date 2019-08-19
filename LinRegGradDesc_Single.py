import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import r2_score

df_train = pd.read_csv('fuel_consumption_train.csv')
df_test = pd.read_csv('fuel_consumption_test.csv')

x_train = df_train['ENGINESIZE']
x_test = df_test['ENGINESIZE']

y_train = df_train['CO2EMISSIONS']
y_test = df_test['CO2EMISSIONS']

x_train = np.transpose(np.array([np.ones(len(x_train)), x_train]))
x_test = np.transpose(np.array([np.ones(len(x_test)), x_test]))

m = len(y_train)

theta = np.array([0.00,0.00])

alpha = 0.01

noOfIter = 1500

i=0

arrCosts = np.zeros(noOfIter)

arrThetas = np.zeros((noOfIter , 2))

while i < noOfIter:  # in every iteration

    # (i) store the theta/parameter values for future comparison in arrThetas
    arrThetas[i][0] = theta[0]
    arrThetas[i][1] = theta[1]

    # (ii) calculate the prediction using existing theta
    # note: dimn(x)=100x2, dimn(h)=100 x 1, hence to calculate
    # 'h',array 'x' should be multipled with matrix of dim(2x1), and
    # that can be achieved by multiplying 'x' with transpose of
    # vector 'theta'
    h = np.dot(x_train, np.transpose(theta))

    # (iii) calculate the cost, i.e., J for given pair of parameters
    # theta[0], theta[1], 'h' which is the predicted output and 'y'
    cost = sum(np.square(h - y_train)) * (1 / (2 * m))

    # (iv) save the value of 'cost' variable in the array arrCosts
    # for future comparison
    # ith index of arrCosts will contain cost value generated in
    # i-th iteration
    arrCosts[i] = cost

    # (v) update values of theta[0] and theta[1] for next iteration
    theta[0] = theta[0] - ((alpha / m) * sum(h - y_train))
    theta[1] = theta[1] - (alpha / m) * sum((h - y_train) * x_train[:, 1])

    # (vi) update the iterator i for continuing the loop
    i = i + 1

lowestCost=min(arrCosts)
temp=np.where(arrCosts==lowestCost)
position=temp[0][0]

intercept = round(arrThetas[position][0],2)
coefficient = round(arrThetas[position][1],2)

# Display the values of the optimum paramaters, i.e. intercept and coefficient
print(f"Optimum Parameters:")
print(f"==>Intercept: {intercept}")
print(f"==>Coeffcient: {coefficient}")
print(f"Linear Regression Model:\n h(x) = {intercept} + ({coefficient})*x")

y_prediction = np.dot(x_test, np.transpose(arrThetas[position]))
x_test_input = x_test[:, 1]
plt.plot(range(noOfIter),arrCosts,color="blue")
plt.title("No. of iterations vs Costs")
plt.xlabel("No. of iterations")
plt.ylabel("J(theta) or cost")
plt.grid(True)
plt.show()

plt.plot(x_test_input , y_test, 'x', color="red")
plt.plot(x_test_input , y_prediction, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("CO2 Emissions")
plt.legend("")
plt.show()

r2Score=r2_score(y_test,y_prediction)
print(f"R Squared = {r2Score}")
