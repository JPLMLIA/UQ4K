'''
This script performs the optimization when all three parameters in Mahdy's
linear example are unknown.
Author        : Mike Stanley
Created       : 03 June 2021
Last Modified : 03 June 2021
'''
import cvxpy as cp
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# for the meb algo
import miniball as mb
from scipy.optimize import differential_evolution

plt.style.use('ggplot')

def center_dist(new_point, center):
    """ Returns distance between new point and center """
    return np.linalg.norm(new_point - center) ** 2

def compute_M_delta(sigma_2, mle_error, df, man_delta=None, conf_level=0.95):
    """ Support chi-sq method and manual delta set """
    if man_delta:
        delta = man_delta
    else:
        gamma = stats.chi2(df=df).ppf(conf_level)  # find the chi-sq quantile
        delta = gamma * sigma_2

    return mle_error + delta

def weight_optimization(S):
    """
    Find dirac weights after min enclosing ball opt
    Parameters:
        S (np arr) : n x m, n - num diracs | m - dim each dirac
    Returns:
        optimized weights over diracs (n,) numpy array
    """
    # find the optimization objects
    n, m = S.shape
    Q_mat = np.zeros(shape=(n, n))

    for t in range(m-1):
        Q_mat += np.outer(qoi_S[:, t], qoi_S[:, t])
        'Assumes both qoi and diracs are passed through'

    v = np.square(S).sum(axis=1)

    # perform the optimization
    p_vec = cp.Variable(n)

    problem = cp.Problem(
        objective=cp.Minimize(
            cp.quad_form(p_vec, Q_mat) - v.T @ p_vec
        ),
        constraints=[
            p_vec >= np.zeros(n),
            cp.sum(p_vec) == 1
        ]
    )

    # solve and check convergence
    problem.solve()
    assert problem.status == 'optimal'

    return p_vec.value

def housing_model(model, theta, x):
    """
    Forward model version of the above model() where the
    data are assumed
    """
    model.get_params()['steps'][1][1].coef_ = theta

    return model.predict(x)

def center_dist_qoi(qoi_func, new_point, center):
    """ Returns distance between new point and center """
    return np.linalg.norm(qoi_func(theta=new_point) - center) ** 2

def qoi_objective_w_penalty(
    new_point,
    model,
    mu,
    center,
    M_delta,
    x_s,
    qoi_func,
    data
):
    """
    Merit function version of the optimization for all parameters unknown.
    Parameters:
        new_point (np arr)
    Returns:
        minimizing objective function criterion
    """
    # find the center distance
    center_dist_term = center_dist_qoi(
        qoi_func=qoi_func,
        new_point=new_point,
        center=center
    )

    # compute the penalty term
    m_theta = housing_model(
        model=model,
        theta=new_point,
        x=x_s
    )
    error = np.linalg.norm(data - m_theta) ** 2
    # print('error:', error)
    merit_term = mu * np.max(np.array([0, error - M_delta]))
    # print('merit:', merit_term)
    return - center_dist_term + merit_term
    
def qoi_min_e_ball(
    model,
    sigma_2,
    data,
    x_s,
    qoi_func,
    epsilon_0=0.0001,
    conf_lev=0.95,
    man_delta=None,
    mu=1e3,
    bounds=[[-30, 30]]*3,
    max_iter=10
):
    """
    Minimal Enclosing ball algorithm for Mahdy's example where all
    parameters are unknown.
    Primary data objects:
    - S         : set of optimized points
    - epsilon_0 : stopping criterion
    - beta      : significance level
    Parameters:
        sigma_2   (float)  : data variance
        data      (np arr) : (n,) data array
        x_s       (np arr) : sampled x values to run forward model
        qoi_func  (np arr) : function evaluated on theta to give qoi
        epsilon_0 (float)  : stopping criterion
        conf_lev  (float)  : confidence level used in chi-sq calc of delta
        man_delta (float)  : manually set delta (default None uses chi-sq calc)
        mu        (float)  : penalty coefficient for merit function
        bounds    (list)   : list of bounds for use in the diff evolution algo
        max_iter  (int)    : max # of iterations of finding boundary points
    Returns:
        mle_theta (np arr) : MLE of parameters given data
        M_delta   (float)  : level-set constraint
        S         (list)   : collection of diracs
        qoi_S     (list)   : collection of points around which min enclosing
                             ball is found in qoi space
        center    (np arr) : converged center
        radius_0  (float)  : converged radius of minimum enclosing ball
    """
    S = []
    d = len(bounds)
    
    
    # find the MLE
    # Can have optimization for this or just use other methods
    # if it matches MLE theoretically
    mle_theta = model.get_params()['steps'][1][1].coef_
    
    # Estimate variance using MLE
    # This overrides the sigma2 passed in the function call
    diff_var = y - housing_model(model,  mle_theta, x_s)
    sigma_2 = np.var(diff_var)
    print(sigma_2)
    
    mle_error = np.square(diff_var).sum()
    
    print('mle_theta:', mle_theta)
    print('mle_error:', mle_error)
    print('Done with MLE')

    # compute M_delta
    M_delta = compute_M_delta(
        sigma_2=sigma_2,
        mle_error=mle_error,
        df=9,
        man_delta=man_delta,
        conf_level=conf_lev
    )
    print('Done with delta:', M_delta)

    # set variables for starting loop
    center = qoi_func(theta=mle_theta).copy()
    S.append(mle_theta.copy())
    radius_0 = 0
    e = 2 * epsilon_0
    i = 0    
    
    while (e >= epsilon_0) & (i < max_iter):

        # update the objective function
        obj_partial = partial(
            qoi_objective_w_penalty,
            model=model,
            mu=mu,
            center=center,
            data=data,
            M_delta=M_delta,
            x_s=x_s,
            qoi_func=qoi_func
        )

        # find boundary point
        de_result = differential_evolution(
            func=obj_partial,
            bounds=bounds
        )
        assert de_result['success']

        # check if new point has larger radius
        if np.linalg.norm(qoi_func(theta=de_result['x']) - center) >= radius_0:
            S.append(de_result['x'])
        qoi_S = [qoi_func(theta=item) for item in S]
        print(qoi_S)
        # find the minimum enclosing ball for S
        C, r2 = mb.get_bounding_ball(np.array(qoi_S))
        center = C

        # update radius change
        e = np.abs(np.sqrt(r2) - radius_0)
        print(r2)
        radius_0 = np.sqrt(r2)
        
        
        # check size of set S -- potentially do not need this
        if len(S) > d + 1:

            # eliminate element in S with smallest distance from c
            distances = [np.linalg.norm(center - S_i) for S_i in qoi_S]
            remove_idx = np.argmin(distances)
            print('removed:', qoi_S[remove_idx])
            del S[remove_idx]
            del qoi_S[remove_idx]
        
        print(i)
        i += 1
    
    
    return mle_theta, M_delta, np.array(S), np.array(qoi_S), center, radius_0
    
if __name__=="__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import Ridge
    
    # Split like this to get consistent results, can use randomization to test
    # further.
    features, target = fetch_california_housing(return_X_y=True)
    f_train = features[:int(len(features)/2)]
    f_test = features[int(len(features)/2):]
    t_train = target[:int(len(target)/2)]
    t_test = target[int(len(target)/2):]
    
    # model needs to be trained in order to edit its coefficients
    degree = 1
    lin_model = make_pipeline(PolynomialFeatures(degree), Ridge(solver='svd'))
    lin_model.fit(f_train, t_train)
    predict = lin_model.predict(f_train)
    
    x_s = f_train
    y = np.array([t_train])
    
    SIGMA2 = 0.1            # noise variance (overridden in method)
    EPSILON = 0.000001      # stopping criterion for odad_min_e_ball algo
    CONF_LEV = 0.95         # 1 - beta_alpha - i.e., prob not violating
    DELTA = 1E3             # set delta, use when desire not to use chi-sq
    MU = 1e-2               # strength of penalty
    BOUNDS = [[-1, 1]]*9    # variable bounds for diff evol algo
    MAX_IT = 40             # controls number of steps in ball algo

    # perform the optimization to find center and radius
    num_test = 5
    intervals = np.zeros((num_test, 2))
    centers = []
    targets = t_test[0:num_test]
    for i in range(num_test):
        # If the model isn't refit every time, the inside MLE line takes the
        # MLE to be one of the boundaries of the previous iteration because
        # the model was run with those coefficients.
        lin_model.fit(f_train, t_train)
        qoi_x = np.array([f_test[i,:]])
        qoi_f = partial(
            housing_model,
            model=lin_model,
            x=qoi_x
        )
        
        mle_theta, M_delta, S, qoi_S, center, radius_0 = qoi_min_e_ball(
            model=lin_model,
            sigma_2=SIGMA2,
            data=y,
            x_s=x_s,
            qoi_func=qoi_f,
            epsilon_0=EPSILON,
            conf_lev=CONF_LEV,
            man_delta=DELTA,
            mu=MU,
            bounds=BOUNDS,
            max_iter=MAX_IT
        )
        
        centers.append(center)
        intervals[i, 0] = center - radius_0
        intervals[i, 1] = center + radius_0
        
        print('----- Center and Radius -----')
        print(qoi_S)
        print(center)
        print(radius_0)
    
    idx = np.argsort(targets)
    test_contains = []
    sorted_intervals = intervals[idx]
    
    for i, t in enumerate(targets[idx]):
        if t <= sorted_intervals[i, 1] and t >= sorted_intervals[i, 0]:
            test_contains.append(1)
        else:
            test_contains.append(0)
    
    p_opt = weight_optimization(S=qoi_S)
    '''
    # perform optimization to find dirac weights
    n, m = S.shape
    Q_mat = np.zeros(shape=(n, n))
    Q_mat += np.outer(qoi_S, qoi_S)
    'Assumes both qoi and diracs are passed through'

    v = np.square(qoi_S)

    # perform the optimization
    p_vec = cp.Variable(n)

    problem = cp.Problem(
        objective=cp.Minimize(
            cp.quad_form(p_vec, Q_mat) - v.T @ p_vec
        ),
        constraints=[
            p_vec >= np.zeros(n),
            cp.sum(p_vec) == 1
        ]
    )

    # solve and check convergence
    problem.solve()
    assert problem.status == 'optimal' # Should be changed to make QOI work
    p_opt = p_vec.value
    print('----- Dirac Weights -----')
    print(p_opt)
    '''
    
    '''
    from cvxopt import matrix, solvers
    Q = matrix(2*Q_mat);
    v_vec = matrix(v);
    G_np = -1*np.eye(n)        ; G = matrix(G_np);
    h_np = np.zeros(n)         ; h = matrix(h_np);
    A_np = np.ones(n)          ; A = matrix(A_np, (1,n));
    b_np = np.array([1.0])     ; b = matrix(b_np);
    sol=solvers.qp(Q, v_vec, G, h, A, b)
    print('x is:', sol['x'])
    '''
