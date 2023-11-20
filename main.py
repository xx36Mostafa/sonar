import pandas as pd
import numpy as np
from sklearn.model_selection import KFold , GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import time

class task:
    def __init__(self):
        self.response = False

    def run(self):
        rangesList = [None,(0,1),(-1,1),(0,9)]
        models = [KNeighborsClassifier(),DecisionTreeClassifier(),SVC(),RandomForestClassifier()]
        for model in models:
            print(f'Model Name Running Now: {model}')
            for i in rangesList:
                if i != None:
                    self.response = True
                    print(f'With Scalling Range {i}') 
                    X,y = self.getXY(i)
                else:
                    self.response = False
                    print('Without Scalling:') 
                    X,y = self.getXY()
                hyper = hyperParams(X,y,model=model)
                self.model,self.kfold,bestParams = hyper.hyperparameter()
                print('The Best Params is',end=' ')
                for key,value in bestParams.items():
                    print(f'{key}:{value}',end=' | ')
                print(end='\n')
                t = time.time()
                self.train(X,y)
                print('CPU Time is:',str(float(time.time()-t))[:7])
            print('*'*100)

    def getXY(self,fRange=None):
        data = pd.read_csv('sonar.csv')
        X,Y = data.iloc[:,:-1].values,data.iloc[:,-1].values
        if self.response:
            scaler = MinMaxScaler(feature_range=fRange)
            X = scaler.fit_transform(X)
        return X,Y
    
    def train(self,X,Y):
        Acc = []
        iteration_number = 0
        for indexTrain,indexTest in self.kfold.split(X):
            iteration_number = iteration_number + 1
            self.model.fit(X[indexTrain],Y[indexTrain])
            print("iteration_number =", iteration_number , '| Accuracy Is >>' , self.model.score(X[indexTest],Y[indexTest]))
            Acc.append(self.model.score(X[indexTest],Y[indexTest]))
        print('The Mean Accuracy Is :',np.average(Acc))

class hyperParams:
    def __init__(self,X,Y,model):
        self.X,self.Y = X,Y
        self.model = model # => Model Use To Train Like : Knn - SVM ....
        self.key = self.get_model_name(model)

    def get_model_name(self, model):
        return str(model.__class__).split('.')[-1][:-2]

    def hyperparameter(self):
        cv = KFold(n_splits=10, shuffle=True)
        knnParams = {
            'KNeighborsClassifier': {
                'n_neighbors': range(2, 20),
                'weights': ['uniform', 'distance']
            },

            'SVC': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
                'kernel': ['linear','rbf']
            },

            'LogisticRegression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2']
            },

            'DecisionTreeClassifier': {
                'criterion': ["gini", "entropy"],
                'splitter': ["best", "random"]
            },
            
            'RandomForestClassifier' : {
                'n_estimators': [10, 50],
                'max_depth': [None, 10,],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
        }
        gridSearch = GridSearchCV(estimator=self.model, param_grid=knnParams[self.key],cv=cv)
        gridSearch.fit(self.X,self.Y)
        model = self.model.set_params(**gridSearch.best_params_)
        return model, cv , gridSearch.best_params_

if __name__ == '__main__':
    run = task()
    run.run()