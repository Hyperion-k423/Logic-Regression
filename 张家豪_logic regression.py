import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math

# load dataset
X_train, y_train = load_data("ex2data1.txt")

print("first of five elements in y_train are :\n",y_train[:5])
print("type of y_train :",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

#Plot examples
plot_data(X_train,y_train[:],pos_label="Admitted",neg_label="Not admitted")
#图例显示为 × 和 ⭕ 的原因，是因为 plot_data 函数中 plt.scatter() 绘制散点图时，分别使用了 marker='o' 和 marker='x' 来表示正类和负类的数据点。因此，图例中的标记符号直接反映了图中数据点的标记样式。

#Set the y-axis label
plt.ylabel('Exam 2 score')
#Set the x-axis label
plt.xlabel('Exam 1 score')
plt.legend(loc="upper right")
plt.show()

# UNQ_C1
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    compute the sigmoid of z
    Args:
        z(ndarray):A scalar,numpy array of any size.
    :return:
        g(ndarray):sigmoid(z),with the same shape as z
    """
    ### START CODE HERE ###
    g = 1 / (1 + np.exp(-z))
    ### END SOLUTION ###

    return g
print("sigmoid(0) = " + str(sigmoid(0)))
#If z is a matrix
print("sigmoid([-1,0,1,2]) = " + str(sigmoid(np.array([-1,0,1,2]))))

# UNIT TESTS
from public_tests import *
sigmoid_test(sigmoid)

# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, lambda_= 1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    m,n = X.shape
    ### START CODE HERE ###
    cost = 0
    for i in range(m):
        z_wb = np.dot(w,X[i]) + b   #这一句放在循环外也可以
        f_wb = sigmoid(z_wb)
        cost += (-y[i]*np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb))     #此处(-y[i]*np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb))即是loss_sum
    total_cost = cost/m
    ### END CODE HERE ###

    return total_cost

m, n = X_train.shape
# Compute and display cost with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.
cost = compute_cost(X_train, y_train, initial_w, initial_b)
print('Cost at initial w (zeros): {:.3f}'.format(cost))

# Compute and display cost with non-zero w
test_w = np.array([0.2,0.2])
test_b = -24.
cost = compute_cost(X_train,y_train,test_w,test_b)
print('Cost at test w,b: {:.3f}'.format(cost))
# UNIT TESTS
compute_cost_test(compute_cost)


# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder.
    Returns
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0
    ### START CODE HERE ###
    for i in range(m):
        # Calculate f_wb (exactly as you did in the compute_cost function above)
        f_wb =sigmoid(np.dot(w,X[i]) + b)

        # Calculate the  gradient for b from this example
        dj_db_i = f_wb - y[i] # Your code here to calculate the error

        # add that to dj_db
        dj_db += dj_db_i

        # get dj_dw for each attribute
        for j in range(n):
            # You code here to calculate the gradient from the i-th example for j-th attribute
            dj_dw_ij =(f_wb - y[i]) * X[i,j]
            dj_dw[j] += dj_dw_ij

        # divide dj_db and dj_dw by total number of examples
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    ### END CODE HERE ###

    return dj_db, dj_dw

# Compute and display gradient with w initialized to zeroes
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w (zeros):{dj_db}' )
print(f'dj_dw at initial w (zeros):{dj_dw.tolist()}' )

# Compute and display cost and gradient with non-zero w
test_w = np.array([ 0.2, -0.5])
test_b = -24
dj_db, dj_dw  = compute_gradient(X_train, y_train, test_w, test_b)

print('dj_db at test_w:', dj_db)
print('dj_dw at test_w:', dj_dw.tolist())
# UNIT TESTS
compute_gradient_test(compute_gradient)


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(X)

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw
        b_in = b_in - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            cost = cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0 or i == (num_iters - 1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w_in, b_in, J_history, w_history  # return w and J,w history for graphing

np.random.seed(1)
intial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
initial_b = -8


# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b,
                                   compute_cost, compute_gradient, alpha, iterations, 0)

#Plotting the decision boundary
plot_decision_boundary(w, b, X_train, y_train)
plt.show()


# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###
    for i in range(m):
        f_wb = sigmoid(np.dot(X[i],w) + b )
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    ### END CODE HERE ###
    return p

# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3
tmp_X = np.random.randn(4, 2) - 0.5
tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')
# UNIT TESTS
predict_test(predict)

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))