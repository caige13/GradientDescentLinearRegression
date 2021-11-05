import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
import os, sys
from datetime import datetime
import matplotlib.pyplot as plt

train_size = .8
test_size = .2
max_iter = 100000
init_learning_rate = .001
learning_rate_method = 'adaptive'

# Starting up logging
logFile = open("Part2_logging.txt", "a")
logFile.write("\n\n***Program Start: " +str(datetime.now())+"***")
logFile.write("\nLearning rate method: "+str(learning_rate_method)+" Max Iteration: "+str(max_iter)+" init_learning_rate: " +str(init_learning_rate))

# Loading in the data with pandas
studentDF = pd.read_csv("https://raw.githubusercontent.com/caige13/GradientDescentLinearRegression/main/student-por.csv", ";")
studentDF = studentDF.dropna()
studentDF = studentDF.drop_duplicates()

# Selecting the Columns to use
X = pd.DataFrame(np.c_[studentDF['G1'], studentDF['G2'], studentDF['studytime'],
                       studentDF['failures'], studentDF['Medu'],studentDF['Fedu'],
                       studentDF['health'], studentDF['traveltime'], studentDF['Dalc'],
                       studentDF['Walc']],
                 columns=['G1','G2', 'studytime', 'failures', 'Medu', 'Fedu', 'health',
                          'traveltime', 'Dalc', 'Walc'])
Y = studentDF['G3']

# Split the data set into training and testing.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size, random_state=5)

# do Gradient Descent linear regression build in with sklearn
lin_model = SGDRegressor(learning_rate=learning_rate_method, max_iter=max_iter, eta0=init_learning_rate)
lin_model.fit(X_train, Y_train)

# get the Y_hat values
Y_train_predict = lin_model.predict(X_train)
Y_test_predict = lin_model.predict(X_test)
r2_test = r2_score(Y_test, Y_test_predict)
r2_train = r2_score(Y_train, Y_train_predict)
rmse_test = (np.sqrt(mean_squared_error(Y_test, Y_test_predict)))
rmse_train = (np.sqrt(mean_squared_error(Y_train, Y_train_predict)))

print("Coefficient Values: "+str(lin_model.coef_))
print("Training MSE:  " + str(mean_squared_error(Y_train, Y_train_predict)))
print("Testing MSE:  " + str(mean_squared_error(Y_test, Y_test_predict)))
print("Training r2: "+str(r2_train)+"\nTesting r2: "+str(r2_test))
print("Test RMSE: "+str(rmse_test))
print("Training RMSE: "+str(rmse_train))

logFile.write("\nCoeff Values: "+str(lin_model.coef_))
logFile.write("\nTraining MSE:  " + str(mean_squared_error(Y_train, Y_train_predict)))
logFile.write("\nTesting MSE:  " + str(mean_squared_error(Y_test, Y_test_predict)))
logFile.write("\nr2_test = "+str(r2_test)+"\nr2_train = "+str(r2_train))
logFile.write("\nTest RMSE: "+str(rmse_test))
logFile.write("\nTraining RMSE: "+str(rmse_train))

logFile.close()

# Setting up variables for scatter plot.
X1 = studentDF['G1']
X2 = studentDF['G2']
X3 = studentDF['studytime']

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()

# Plotting various graphs
plt.scatter(Y,X1,X2,X3)
plt.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_predict), max(Y_train_predict)], color='red')
plt.ylabel('Y_train_pred(used LinRegress Formula)')
plt.xlabel('X_train')
plt.title('Linear Regression Result on Training Data')
plt.savefig(os.path.join(sys.path[0], "Part2_Graphs/TrainingRegressionResult.png"))
plt.show()

plt.scatter(Y,X1,X2,X3)
plt.plot([min(X_test[:,1]), max(X_test[:,1])], [min(Y_train_predict), max(Y_train_predict)], color='red')
plt.ylabel('Y_test_pred(used LinRegress Formula)')
plt.xlabel('X_test')
plt.title('Linear Regression Result on Test Data')
plt.savefig("Part2_Graphs/TestingRegressionResult.png")
plt.show()

ci = 1.96 * np.std(Y_train_predict)/np.mean(Y_train_predict)
fig,ax = plt.subplots()
ax.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_predict), max(Y_train_predict)], color='red')
ax.set_title('Regression line with Confidence Bands (G1)')
ax.set_ylabel('Y_train_pred')
ax.set_xlabel('X_train(G1)')
ax.fill_between(X_train[:,0], (Y_train_predict-ci), (Y_train_predict+ci), alpha=.2, color='tab:orange')
plt.savefig("Part2_Graphs/TrainDataConfidenceBandsG1.png")
plt.show()

fig,ax = plt.subplots()
ax.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_predict), max(Y_train_predict)], color='red')
ax.set_title('Regression line with Confidence Bands (G2)')
ax.set_ylabel('Y_train_pred')
ax.set_xlabel('X_train(G2)')
ax.fill_between(X_train[:,1], (Y_train_predict-ci), (Y_train_predict+ci), alpha=.2, color='tab:orange')
plt.savefig("Part2_Graphs/TrainDataConfidenceBandsG2.png")
plt.show()
plt.close()