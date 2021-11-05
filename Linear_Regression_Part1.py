import distutils.util
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from GradientDescentLinearRegressionNoLib import GradientDescentLinearRegressionNoLib
from datetime import datetime
from sklearn.metrics import r2_score

train_size = .8
test_size = .2

# Handles Verbose actions because it can be computation heavy.
verbose = input("Do you want the linear regression process to be verbose?\n" +
                "(this gives a step by step process, basic logging is provided by default)"+
                "\nAnswer(True/False): ")
verbose = distutils.util.strtobool(verbose)

# Opening up logging file for initial logging
logFile = open("Part1_logging.txt", "a")
logFile.write("\n\n\n***Start Program: "+str(datetime.now())+"***\ntrain size: "+str(train_size)+"\ttest_size: "+str(test_size))

# Pulling in the data and loading it into pandas DF also cleaning up data
#studentDF = pd.read_csv("C:\\Dev\\WhiteWineML\\student-por.csv", ";")
studentDF = pd.read_csv("https://raw.githubusercontent.com/caige13/GradientDescentLinearRegression/main/student-por.csv", ";")
studentDF = studentDF.dropna()
studentDF = studentDF.drop_duplicates()
print(studentDF)

# selecting data used for training the model. X: features Y: Target
X = pd.DataFrame(np.c_[studentDF['G1'], studentDF['G2'], studentDF['studytime'],
                       studentDF['failures'], studentDF['Medu'],studentDF['Fedu'],
                       studentDF['health'], studentDF['traveltime'], studentDF['Dalc'],
                       studentDF['Walc']],
                 columns=['G1','G2', 'studytime', 'failures', 'Medu', 'Fedu', 'health',
                          'traveltime', 'Dalc', 'Walc'])
Y = studentDF['G3']

logFile.write("\nColumns used: " + str(X.columns.values))

# Split the Data into 80/20 for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, test_size=test_size, random_state=5)
logFile.write("\n\t\tShapes\nX_train: " + str(X_train.shape) + "\nY_train: " + str(Y_train.shape) +
              "\nX_test: " + str(X_test.shape) + "\nY_test: " + str(Y_test.shape))

# Populate the first column with 1 to give same shape as Coeff
X_0 = []
for i in range( 0, X_train.shape[0]):
    X_0.append(1)
X_train.insert(0,"X0", X_0, True)

# Convert the Pandas DFs into a numpy array for convenience
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()

# Manually done Gradient Descent
print("Executing Gradient Descent Linear Regression...")
regression = GradientDescentLinearRegressionNoLib(max_iterations=100000, learning_rate=.003)
logFile.close()
regression = regression.fit(X_train, Y_train,logging=True,verbose=bool(verbose), fileName="Part1_logging.txt")

# After Gradient Descent is done
logFile = open("Part1_logging.txt", "a")

# calculating the Y_pred using test data
Y_test_pred = regression.theta[0]
for i in range(1, regression.theta.shape[0]):
    temp = X_test[:,i-1]*regression.theta[i]  # dJ/d theta_i.
    Y_test_pred = Y_test_pred+temp

# Calculating the Y_pred using training data
Y_train_pred = regression.theta[0]
for i in range(1, regression.theta.shape[0]):
    temp = X_train[:,i-1]*regression.theta[i]
    Y_train_pred = Y_train_pred+temp

# Calculate R^2
r2_test = r2_score(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)

# calculate the MSE of the Test data with the linear model
MSE_test = np.mean((Y_test - Y_test_pred) ** 2)
r2_test = r2_score(Y_test, Y_test_pred)
r2_train = r2_score(Y_train, Y_train_pred)
rmse_test = np.sqrt(MSE_test)
rmse_train = np.sqrt(regression.cost_hist[-1])

logFile.write("\nFinal Coeff: " + str(regression.theta))
logFile.write("\nFinal Cost(MSE) Value: " + str(regression.cost_hist[-1]))
logFile.write("\nTest Data MSE: "+str(MSE_test))
logFile.write("\nr2_test = "+str(r2_test)+"\nr2_train = "+str(r2_train))
logFile.write("\nTest RMSE: "+str(rmse_test))
logFile.write("\nTraining RMSE: "+str(rmse_train))
logFile.close()

print("Final Cost(MSE) Value: " + str(regression.cost_hist[-1]))
print("Test Data MSE: "+str(MSE_test))
print("Coefficient Values: "+str(regression.theta))
print("Training r2: "+str(r2_train)+"\nTesting r2: "+str(r2_test))
print("Test RMSE: "+str(rmse_test))
print("Training RMSE: "+str(rmse_train))
# Graph the data in a Scatter plot
X1 = studentDF['G1']
X2 = studentDF['G2']
X3 = studentDF['studytime']

# G1, G2 are the 2 most important ones. The others have skewed values that dont graph well.
plt.scatter(Y,X1, X2)
plt.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_pred), max(Y_train_pred)], color='red')
plt.ylabel('Y_train_pred(used LinRegress Formula)')
plt.xlabel('X_train')
plt.title('Linear Regression Result on Training Data')
plt.savefig("Part1_Graphs/TrainRegressionLine.png")
plt.show()

plt.scatter(Y,X1, X2)
plt.plot([min(X_test[:,1]), max(X_test[:,1])], [min(Y_test_pred), max(Y_test_pred)], color='red')
plt.ylabel('Y_test_pred(used LinRegress Formula)')
plt.xlabel('X_test')
plt.title('Linear Regression Result on Test Data')
plt.savefig("Part1_Graphs/TestRegressionLine.png")
plt.show()

# Plotting the G3 vs G3 with Y_pred
plt.scatter(Y,Y)
plt.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_pred), max(Y_train_pred)], color='red')
plt.ylabel('G3(Final Grade/Target)')
plt.xlabel('G3(Final Grade/Target)')
plt.title('G3 vs G3 With Training Data Results')
plt.savefig("Part1_Graphs/GraphY_predOnG3vsG3")
plt.show()

# plotting regression line with confidence band of G1 VS Y_train_pred
ci = 1.96 * np.std(Y_train_pred)/np.mean(Y_train_pred)
fig,ax = plt.subplots()
ax.plot([min(X_train[:,1]), max(X_train[:,1])], [min(Y_train_pred), max(Y_train_pred)], color='red')
ax.set_title('Regression line with Confidence Bands (G1)')
ax.set_ylabel('Y_train_pred')
ax.set_xlabel('X_train(G1)')
ax.fill_between(X_train[:,1], (Y_train_pred-ci), (Y_train_pred+ci), alpha=.2, color='tab:orange')
plt.savefig("Part1_Graphs/TrainDataConfidenceBandsG1.png")
plt.show()

# plotting regression line with confidence band of G2 VS Y_train_pred
fig,ax = plt.subplots()
ax.plot([min(X_train[:,1]), max(X_train[:,2])], [min(Y_train_pred), max(Y_train_pred)], color='red')
ax.set_title('Regression line with Confidence Bands (G2)')
ax.set_ylabel('Y_train_pred')
ax.set_xlabel('X_train(G2)')
ax.fill_between(X_train[:,2], (Y_train_pred-ci), (Y_train_pred+ci), alpha=.2, color='tab:orange')
plt.savefig("Part1_Graphs/TrainDataConfidenceBandsG2.png")
plt.show()
plt.close()

# Lets you review data in log to make a conclusion
logFile = open("Part1_logging.txt", "a")
conclusion = input("Review log; What is your conclusion: ")
logFile.write("\nCONCLUSION: "+conclusion)

logFile.close()