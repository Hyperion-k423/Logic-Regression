#线性可分
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path='ex2data1.txt'
data=pd.read_csv(path,names=['Exam 1','Exam 2','Accepted'])
data.head()
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1]['Exam 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='Exam 1',ylabel='Exam 2')
plt.show()
def get_Xy(data):
    data.insert(0,'one',1)
    X_=data.iloc[:,:-1]
    X_=X_.values
    y_=data.iloc[:,-1]
    y=y_.values.reshape(len(y_),1)
    return X_,y
X,y=get_Xy(data)
X.shape
y.shape
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costFuncation(X,y,theta):
    A=sigmoid(X@theta)
    first=y*np.log(A)
    second=y*np(1-y)*np.log(1-A)
    return -np.sum(first+second)/len(X)
theta=np.zeros((3,1))
theta.shape
cost_init=costFuncation(X,y,theta)
print(cost_init)
def gradient_descent(X,y,theta,iters,alpha):
    m=len(X)
    cost=[]
    for i in range(iters):
        A=sigmoid(X@theta)
        theta=theta-(alpha/m)*X.T@(A-y)
        cost=costFuncation(X,y,theta)
        cost.append(cost)
        if i % 1000 == 0:
            print(cost)
            return theta,cost
alpha=0.004
iters=200000
cost,theta_final=gradient_descent(X,y,theta,iters,alpha)
def predict(X,theta):
    prob=sigmoid(X@theta)
    return [1 if x>=0.5 else 0 for x in prob]
y_=np.array(predict(X,theta_final))
y_pre=y_reshape(len(y_),1)
acc=np.mean(y_pre==y_)
print(acc)
coef1=-theta_final[0,0]/theta_final[2,0]
coef2=-theta_final[1,0]/theta_final[2,0]
fig,ax=plt.subplots()
ax.scatter(data[data['Accepted']==0]['Exam 1'],data[data['Accepted']==0]['Exam 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Exam 1'],data[data['Accepted']==1['Exam 2'],c='r',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='Exam 1',ylabel='Exam 2')
ax.plot(x,y,c='g')
plt.show()
#线性不可分
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path='ex2data2.txt'
data=pd.read_csv(path,names=['Test 1','Test 2','Accepted'])
data.head()
fig, ax = plt.subplots()
ax.scatter(data[data['Accepted']==0]['Test 1'],data[data['Accepted']==0]['Test 2'],c='r',marker='x',label='y=0')
ax.scatter(data[data['Accepted']==1]['Test 1'],data[data['Accepted']==1]['Test 2'],c='b',marker='o',label='y=1')
ax.legend()
ax.set(xlabel='Test 1',ylabel='Test 2')
plt.show()
def feature_mapping(x1,x2,power):
    data={}
    for i in np.range(power+1):
        for j in np.arange(+1):
            data=['F{}{}'.format(i-j,j)*np.power(x2,j)]
    return pd.DataFrame(data)
x1=data['Test 1']
x2=data['Test 2']
data2=feature_mapping(x1,x2,6)
X=data2.values
X.shape
y=data2.iloc[:,-1].values
y=y.reshape(len(y),1)
y.shape
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costFunction(X,y,theta,lr):
    A=sigmoid(X@theta)
    first=y*np.log(A)
    second=y*(1-y)*np.log(1-A)
    reg=np.sum(np.power(theta[1:],2))*(lr/(2*len(X)))
    return -np.sum(first+second)/len(X)+reg
theta=np.zeros((28,1))
theta.shape
lamda=1
cost_init=costFunction(X,y,theta,lamda)
print(cost_init)
def gradient_descent(X,y,theta,alpha,iters,lamda):
    cost=[]
    for i in range(iters):
        reg=theta[1:]*(lamda/len(X)
        reg=np.inert(reg,)
        theta=theta-(X.T@(sigmoid(X@theta)-y))*alpha/len(X)
        cost=costFunction(X,y,theta)
        if i % 1000 == 0:
            print(cost)
            return theta,cost
alpha=0.001
iters=200000
lamda=0.001
theta_final,costs=gradientDescent(X,y,theta,alpha,iters,lamda)
def predict(X,theta):
    prob=sigmoid(X@theta)
    return [1 if x>=0.5 else 0 for x in prob]
y_=np.array(predict(X,theta_final))
y_pre=y_.reshape(len(y_),1)
acc=np.mean(y_pre==y_)
print(acc)
x=np.linspace(-1.2,1.2,200)
xx,yy=np.meshgrid(x,x)
z=feature_mapping(xx.ravel(),yy.ravel(),6)values
zz=z@theta_final.reshape
zz=zz.reshape(xx.shape)
plt.contourf(xx,yy,zz,0)
plt.show()

















