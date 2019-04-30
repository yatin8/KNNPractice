import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dfx=pd.read_csv('./knnData/xdata.csv')
dfy=pd.read_csv('./knnData/ydata.csv')
X=dfx.values
Y=dfy.values
# Y[100][1]=5 #MULTIPLE COLORS OCCURED IN PLOT DUE TO MULTIPLE VALUE OF Y(intially only 0and1)
# Y[101][1]=8
x_train=X[:,1:]
y_train=Y[:,1:]


def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x_train,y_train,x_test,k=5):
    dist_val=[]
    m=x_train.shape[0]
    for ix in range(m):
        d=distance(x_test,x_train[ix])
        dist_val.append([d,y_train[ix]])

    dist_val=sorted(dist_val)
    dist_val=dist_val[:k]

    y=np.array(dist_val)
    ans=np.unique(y[:,1],return_counts=True)
    index=ans[1].argmax()
    prediction=ans[0][index]
    return int(prediction)

x_test=np.array([3,2])
prediction=knn(x_train,y_train,x_test)
print(prediction)

plt.style.use('seaborn')
plt.scatter(X[:,1],X[:,2],c=Y[:,1])
plt.scatter(x_test[0],x_test[1],color='red')
plt.show()
