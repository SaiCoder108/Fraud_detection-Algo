import numpy as np # Math array
from sklearn.tree import DecisionTreeRegressor # Decision tree



temp = np.array([[-5], [0], [10], [20], [25], [30]]) # Feature input for each hour (in this case tempreature)
actual = np.array([1300, 1200, 1100, 900, 900, 900]) # The output we want the model to predict based on the input

lr = 0.5 # Learing rate (higher means correction has more influence)

pred = np.full(len(actual), actual.mean()) # Takes the mean of actual and makes a list named pred which is the length of actual (6)
for n in range(1, 3): # Decides the number of iterations/trees -> This value sets n_estimators
    residual = actual - pred # Subtracts pred from actuals (array-array) which outputs the delta of our pred vs the desired output
    tree = DecisionTreeRegressor(max_depth=1).fit(temp, residual) # inputs the tempreature inputs and residuasl to coralate them (max depth means only 1 yes or no)
    pred = pred + lr * tree.predict(temp) # output the learned state and apply it to the prediction 
    print(f"tree {n}", pred, "MSE", round(((actual - pred) ** 2).mean(), 1))
