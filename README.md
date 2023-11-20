# sonar
This data determines if the radiation coming from Mine Or Rock
This script uses five different models
1- KNeighborsClassifier
2- SVC
3- DecisionTreeClassifier
4- RandomForestClassifier
5- LogisticRegression
It divides the data using KFold , and set the n splits to 10
We do training in four different ways
1- without sciling 
2- with sciling in range (0,1)
3- with sciling in range (-1,1)
4- with sciling in range (0,9)
First, it performs a loop on the list where the models are located and uses GridSearch To Find Best hyperParameters
Then it divides the data using keyfold and calculates the Accuracy in each iteration and then calculates the average Accuracy.
