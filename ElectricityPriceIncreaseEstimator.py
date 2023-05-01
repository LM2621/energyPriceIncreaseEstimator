import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

energyDf = pd.read_csv("./electricityprices.csv")


# create an instance of the model
linReg = LinearRegression()

targetVar = energyDf.priceIncrease
energyDf = energyDf.drop("priceIncrease" , axis=1)
energyDf = energyDf.drop("Country" , axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, resultActual = train_test_split(energyDf,
                                                          targetVar,
                                                          test_size=1/10)
#Linear regression
linReg.fit(X_train,y_train);
resultLinReg = linReg.predict(X_test)

#Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
resultRf = rf.predict(X_test)

# Decision tree regression model
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
dt.fit(X_train, y_train)

# Make predictions on the test data
resultDT = dt.predict(X_test)

# Create a gradient boosting regression model
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=41)

# Train the model on the training data
gb.fit(X_train, y_train)

# Make predictions on the test data
resultsGB = gb.predict(X_test)

# Create a Ridge regression model with alpha = 0.1
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=0.1)

# Train the model on the training data
ridge.fit(X_train, y_train)

# Make predictions on the test data
resultsRegularizedRidge = ridge.predict(X_test)

#Create voting regressor with above models
from sklearn.ensemble import VotingRegressor
voting_reg = VotingRegressor(estimators=[('lr', linReg), ('dt', dt), ('gb', gb), ('rf', rf)])

# Fit the Voting Regressor on the training data
voting_reg.fit(X_train, y_train)

# Predict on the test data using the Voting Regressor
resultsVR = voting_reg.predict(X_test)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(resultActual, resultDT)
print("Score of decision tree:")
print(mae)

mae = mean_absolute_error(resultActual, resultLinReg)
print("Score of linear regression:")
print(mae)

mae = mean_absolute_error(resultActual, resultRf)
print("Score of random forest:")
print(mae)

mae = mean_absolute_error(resultActual, resultsGB)
print("Score of gradient boost:")
print(mae)

mae = mean_absolute_error(resultActual, resultsRegularizedRidge)
print("Score of regularized bridge:")
print(mae)

mae = mean_absolute_error(resultActual, resultsVR)
print("Score of voting regression:")
print(mae)

print("done")