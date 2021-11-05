# GradientDescentLinearRegression
## Overview:
Part 1 and Part 2 are seperate programs ran seperately. The GradientDescentLinearRegressionNoLib file
is the file that contains the algorithm for gradient descent. The class is designed to support other
types of algorithms for the learning rate such as adaptive in the future if I choose to add them. Hints
why there is a Method input for fit and there is only constant learning rate method.

I suggest creating a virtual environment with py -m venv .venv command and the going into that .venv
file and installing the libraries with the requirements.txt file. The command to install from a 
requirements.txt file is pip install -r [path to requirements.txt]


## Part 1:
The .csv files are hosted on my github repository, but I will include them also in the zipped up version
for easier access. The program is only dependent upon location for the logging files and the graphs

Location to save Graphs is in the folder labeled Part1_Graphs.
Location for logging file is in the same folder as the executing script.

Instructions
Execute the part 1 script and either type true or false for verbose.
Verbose: will print to the cmdline step by step the values of Theta, Gradient, MSE Cost, and more info.
This can be a lot of info because it can go up to 100K iterations by default depending on the situation.

It will print the the screen a summary of the dataset. Then it will go into executing the algorithm.
Note this can take some time depending upon the parameters, but if you leave them the way I set it it can
take awhile. Will likly take longer with verbose turned on due to sending info to the buffer to print to cmd.

The program will then prompt you to input your conclusion, and the intention here is to go to the log file
and view the results then input conclusion. The logging file has many previous attempts starting fresh to 
an optimized solution if you wish to walk through the logging file. I provided my answer to the question
in the logging file.

## Part2:
Part 2 is a lot easier, so you should just execute the part 2 script and it will not require any input.
Similar situation I walked through many different parameter setting including different methods to do a 
learning step inside the loggin file for part 2. I provided my answer to the question in the Logging file 2.

I will note that the results can be very diverse using the 3rd part library and is never consistent except with
the adaptive method of doing the learning rate.