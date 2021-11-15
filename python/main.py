import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
def read_data():
    d=pd.read_csv("iris.csv")
    array=d.values
    return array
def train():
    array=read_data()
    x = array[:,1:5]
    y = array[:,5]
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
    X_train=X_train.T
    X_train=np.vstack((X_train,Y_train))
    np.savetxt('train.csv', X_train.T, delimiter=',',fmt='%s')
    np.savetxt('test.csv',X_test, delimiter=',',fmt='%s')
    np.savetxt('test_verify.csv',Y_test,delimiter='/n',fmt='%s')
if __name__ == '__main__':
    train()
