import numpy as np


class GradientDescentLinearRegressionNoLib:

    """
        Initialize the various parameters for gradient based linear regression.

        eps - is the constant value that represents a small enough value to stop at. This number
        should be very small since we are trying to get as close to 0 as possible.

        max_iterations - number of times gradient descent is applied to get best results

        Learning rate - Step size
    """
    def __init__(self, learning_rate=0.001, max_iterations=100000, eps=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.eps = eps

    # prediction array will be shape [n, 1] where n is number of samples.
    def predict(self, X):
        return np.dot(X, self.theta.T)

    # returns the calculations for a cost function.
    def cost(self, X, Y):
        Y_hat = self.predict(X)
        return np.mean((Y - Y_hat) ** 2)

    # Returns the gradient part of formula that must be multiplied bY the step size.
    def gradient(self, X, Y):
        Y_hat = self.predict(X)
        gradient = np.array([])
        for i in range(0, X.shape[1]):
            d_i = 2*sum(X[:, i].reshape(-1,1) * (Y_hat - Y).reshape(-1, 1))  # dJ/d theta_i.
            gradient = np.append(gradient, d_i)
        return gradient / X.shape[0]


    """
        fit is the "driver" function that should be called. 
        fit will execute the gradient descent algorithm to the specifications
        of the parameters set at the initialization of this object. 
        
        X - Pandas Dataframe matrix of shape [n_sampes, n_predictors]
        
        Y - Pandas Dataframe matrix for the target value of shape [n_samples,1]
        
        method - the motivation behind this is to come back and add other ways to
                involve step count into the algorithm like an adaptive step count.
                Vanilla model only supports a standard strategy.
                
        verbose - Used to turn on logging to the shell for more information about
                the process with parameters.
                
        logging - Used for the log file. Only turn on if you have a global variable named
                logFile.
        
        fileName - the name of the text file, note the root will be in the same file 
                as this one.
    """
    def fit(self, X, Y, method="constant", fileName="GradDescentLinRegression.txt", logging=False, verbose=True):
        self.theta = np.zeros(X.shape[1])  # Initialization of params.
        theta_hist = [self.theta]  # History of params.
        cost_hist = [self.cost(X, Y)]  # History of cost.
        if logging:
            logFile = open(fileName, "a")
            logFile.write("\n\nBeginning of Gradient Descent Algorithm.")
            logFile.write("\nalpha(learning rate): " + str(self.learning_rate)+
                          "   MaxIter: "+str(self.max_iterations)+
                          "  Cut off amount: " + str(self.eps))
        for iter in range(self.max_iterations):
            gradient = self.gradient(X, Y)  # Calculate the gradient.
            if method == "constant":
                step = self.learning_rate * gradient  # Calculate standard gradient step.
            else:
                raise ValueError("Method not supported.")
            self.theta = self.theta - step  # Update parameters.
            theta_hist.append(self.theta)  # Save to history.

            # Need to get new prediction with new coeff.
            MSE = self.cost(X, Y)  # Calculate the cost.
            cost_hist.append(MSE)  # Save to history.

            if verbose:
                print("Iter: "+str(iter)+", gradient: "+str(gradient)+", params: "+str(self.theta)+", MSE: "+str(MSE))
            # Stop if update is small enough.
            if np.linalg.norm(theta_hist[-1] - theta_hist[-2]) < self.eps:
                if logging:
                    logFile.write("\nVery low change in Coeff thus stopping early.")
                break

        # Final updates before finishing.
        self.iterations = iter + 1  # Due to zero-based indexing.
        self.theta_hist = theta_hist
        self.cost_hist = cost_hist
        self.method = method
        if logging:
            logFile.close()
        return self