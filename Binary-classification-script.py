import pandas as pd 
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score,
                            confusion_matrix,
                            classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


class Model:
    def __init__(self, datafile = "training.csv"):
        self.df = pd.read_csv(datafile,delimiter=';')
        self.df = self.df.drop(columns=['variable17'],axis=1)
        self.tree = DecisionTreeClassifier()

    def train_test(self):
        self.X,self.y = self.df.drop(['classLabel'],axis=1),self.df['classLabel']

    def split(self, test_size):
        self.train_test()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = test_size, random_state = 42)

    def fit(self):
        self.model = self.tree.fit(self.X_train, self.y_train)

    def predict(self):
        result = self.tree.predict(self.X_test)
        return result
    
    def obj_to_float(self,columns):
        for i in columns:
            self.df[f'variable{i}'] = self.df[f'variable{i}'].str.replace(',','.').astype('float64')

    def fill_na(self):
        self.df.fillna(method='ffill',inplace=True)
        self.df.variable18 = self.df.variable18.fillna(method='bfill')
    
    def cat_to_numeric(self):
        lables = {}
        for col in self.df.columns:
            if self.df[col].dtype=='O':
                lables[col] = LabelEncoder()
                self.df[col]=lables[col].fit_transform(self.df[col])
        return lables

if __name__ == '__main__':
    print(f"{'#'*20} Model uses training Data {'#'*20}")
    model_instance = Model()
    model_instance.obj_to_float([2,3,8])
    model_instance.fill_na()
    labels = model_instance.cat_to_numeric()
    model_instance.split(0.2)
    cls = model_instance.tree.fit(model_instance.X_train,model_instance.y_train)
    y_pred = cls.predict(model_instance.X_test)
    print(classification_report(model_instance.y_test,y_pred))
    print(f"Accurcy Score :{accuracy_score(model_instance.y_test,y_pred)}")
    print("Confusin matrix: \n",confusion_matrix(model_instance.y_test,y_pred))