import numpy as np
import cvxpy as cvx
import time
from matplotlib import pyplot as plt

def shrink(y, v):
    yshape = y.shape
    x = np.zeros(shape=yshape[0], dtype=np.float64)
    p = np.abs(y) > v
    x[p] = y[p] - v * np.sign(y[p])
    return x

def partial_all(x, y, A):
    return 2 * np.dot(np.transpose(A), np.dot(A, x) - y)

def Obj(y, x, A, Lambda):
    return np.square(np.linalg.norm(y - np.dot(A, x), ord=2)) + Lambda * np.linalg.norm(x, ord=1)

def ADMM(y, A, Lambda, err):
    Ashape = A.shape
    x = np.random.random(size=Ashape[1])
    z = np.random.random(size=Ashape[1])
    gamma = np.random.random(size=Ashape[1])

    mu = 100
    pars0 = 2 * np.dot(np.transpose(A), A) + np.eye(Ashape[1]) / mu
    pars1 = np.linalg.inv(pars0)
    pars2 = 2 * np.dot(np.transpose(A), y)

    res = Obj(y, x, A, Lambda)

    while True:
        z = shrink(x - mu * gamma, Lambda * mu)
        x = np.dot(pars1, pars2 + z / mu + gamma)
        gamma += (z - x) / mu
        res_old = res
        res = Obj(y, x, A, Lambda)
        print(res)
        if res == 0:
            break
        rat = np.abs(res - res_old) / res_old
        if rat < err:
            break

    return x

def FSSA(y, A, Lambda): # unfinished
    Ashape = A.shape
    x = np.zeros(shape=Ashape[1], dtype=np.float64)
    theta = np.zeros(shape=Ashape[1], dtype=np.float64)
    active_set = list()

    while True:
        p = (x == 0)
        partialx = partial_all(x, y, A)
        partialx_abs = np.abs(partialx)
        partialx_abs[not p] = 0
        q = partialx_abs > Lambda
        if not q.any():
            break
        max_index = partialx_abs.argmax()
        if partialx[max_index] > Lambda:
            theta[max_index] = -1
        else:
            theta[max_index] = 1
        active_set.append(max_index)
        while True:
            A_a = A[:, active_set]
            x_a = x[active_set]
            theta_a = theta[active_set]
            par1 = np.linalg.inv(np.dot(np.transpose(A_a), A_a))
            par2 = np.dot(np.transpose(A_a), y)
            xanew = np.dot(par1, par2 - theta_a * Lambda / 2)
            # unfinished

def cvx_solve(y, A, Lambda):
    Ashape = A.shape
    x = cvx.Variable(Ashape[1])
    objective = cvx.Minimize(cvx.sum_squares(y - A*x)+Lambda*cvx.norm(x,1))
    prob = cvx.Problem(objective)
    result = prob.solve()
    return x.value

def sparse_array(dim, frac):
    sparse = np.zeros(dim)
    for i in range(len(sparse)):
        if np.random.random()<frac:
            sparse[i] = np.random.random()*2-1.0
    return sparse

if __name__ == "__main__":
    # solve problem min |y-Ax|+Lambda|x|
    leny = 128
    lenx = 256
    Lambda = 0.05
    err = 1e-8
    x_real = sparse_array(lenx, 0.05)
    # Ax = y
    A = np.random.random(size=(leny, lenx))
    for i_A in range(lenx):
        norm_Ai = np.linalg.norm(A[:, i_A])
        A[:, i_A] = A[:, i_A] / norm_Ai
    y = np.dot(A, x_real)

    tic = time.time()
    x_ADMM = ADMM(y, A, Lambda, err)
    t_ADMM = time.time() - tic
    tic = time.time()
    x_cvx = cvx_solve(y, A, Lambda)
    t_cvx = time.time() - tic
    tic = time.time()
    print('obj_real:', Obj(y, x_real, A, Lambda))
    print('obj_ADMM:', Obj(y, x_ADMM, A, Lambda), ' time:', t_ADMM)
    print('obj_cvx:', Obj(y, x_cvx, A, Lambda), ' time:', t_cvx)
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(3, 1, 1)
    ax2 = fig1.add_subplot(3, 1, 2)
    ax3 = fig1.add_subplot(3, 1, 3)
    ax1.plot(list(range(len(x_real))), x_real)
    ax2.plot(list(range(len(x_ADMM))), x_ADMM)
    ax3.plot(list(range(len(x_cvx))), x_cvx)
    ax1.set_ylim([-1.0, 1.0])
    ax2.set_ylim([-1.0, 1.0])
    ax3.set_ylim([-1.0, 1.0])
    ax1.set_title('Real Solution')
    ax2.set_title('ADMM Solution in %.4f s' %t_ADMM)
    ax3.set_title('cvx Solution in %.4f s' %t_cvx)
    plt.show()