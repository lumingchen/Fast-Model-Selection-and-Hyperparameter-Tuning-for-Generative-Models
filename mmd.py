import numpy as np
import statsmodels.stats.multitest
import scipy
import cupy as cp
 
_eps=1.0e-8

def kernelwidthPair(x1, x2):
    '''
    Implementation of the median heuristic. See Gretton 2012
    Pick sigma such that the exponent of exp(- ||x-y|| / (2*sigma2)),
    in other words ||x-y|| / (2*sigma2),  equals 1 for the median distance x
    and y of all distances between points from both data sets X and Y.
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    # print((x1*x1).shape)
    # print((x1*x1)[0][0])
    k1 = np.sum((x1*x1), 1)
    # print((x2*x2).shape)
    # print((x2*x2)[0][0])
    k2 = np.sum((x2*x2), 1)
    
    h = k1[:, None] + k2[None, :]
    del k1, k2
    
    # The norm
    h = h - 2*np.dot(x1, x2.transpose())
#     h = np.array(h, dtype=float)
    
    mdist = np.median(h)
    
    sigma = np.sqrt(mdist/2.0)
    if not sigma: sigma = 1
    
    return sigma


def grbf(x1, x2, sigma):
    '''
    Calculates the Gaussian radial base function kernel
    '''
    n, nfeatures = x1.shape
    m, mfeatures = x2.shape
    
    x1_cp = cp.array(x1)
    x2_cp = cp.array(x2)
    
    k1 = cp.sum((x1_cp*x1_cp), 1)
    q = cp.tile(k1, (m, 1)).transpose()
    del k1  
    
    k2 = cp.sum((x2_cp*x2_cp), 1)
    r = cp.tile(k2.T, (n, 1))
    del k2
    
    h = q + r
    del q,r

    # The norm 
    h = h - 2 * (cp.dot(x1_cp, x2_cp.transpose()))
    return (cp.exp(-1.*h/(2.*pow(sigma,2)))).get()


def MMD(X,Y,sigma=-1):
    '''
    Computes MMD^2 between two samples.
    '''

    if(sigma<0):
        siz=np.min((1000,X.shape[0]))
        sigma1=kernelwidthPair1(X[0:siz],Y[0:siz]);

    Kyy = grbf(Y,Y,sigma)
    Kxy = grbf(X,Y,sigma)
    Kyynd = Kyy-np.diag(np.diagonal(Kyy))
    m = Kxy.shape[0];
    n = Kyy.shape[0];
  
    u_yy=np.sum(Kyynd)*( 1./(n*(n-1)) )
    u_xy=np.sum(Kxy)/(m*n)
    
    Kxx = grbf(X,X,sigma)
    Kxxnd = Kxx-np.diag(np.diagonal(Kxx))
    u_xx=np.sum(Kxxnd)*( 1./(m*(m-1)) )
    MMDXY=u_xx+u_yy-2.*u_xy
#     MMDXY=u_yy-2.*u_xy

    return MMDXY,sigma


def MMD_multi(X,Y,sigma=-1):
    '''
    Computes MMD^2 between a reference distribution and a list of comparing distributions.
    
    Args
    ----
    - X: sample from the reference distribution
    - Y: a list of samples from comparing distributions
    
    '''

    if(sigma<0):
        #Similar heuristics
        siz=np.min((1000,X.shape[0]))
        sigma_list = []
        for y in Y:
            sigma_list.append(kernelwidthPair(X[0:siz],y[0:siz]))
        sigma = np.mean(sigma_list)

    
    Kxx = grbf(X,X,sigma)
    Kxxnd = Kxx-np.diag(np.diagonal(Kxx))
    m = Kxx.shape[0];
    u_xx=np.sum(Kxxnd)*( 1./(m*(m-1)) )
    MMDXY = []
    
    for y in Y:     
        Kyy = grbf(y,y,sigma)
        Kxy = grbf(X,y,sigma)
        Kyynd = Kyy-np.diag(np.diagonal(Kyy))
        
        n = Kyy.shape[0];

        u_yy=np.sum(Kyynd)*( 1./(n*(n-1)) )
        u_xy=np.sum(Kxy)/(m*n)
        MMDXY.append(u_xx+u_yy-2.*u_xy)

    return MMDXY


def Kernels_multi(X, Y, sigma=-1):
    '''
    Compute kernel matrices between the reference data and a list of dataset.
    '''

    if(sigma<0):
        siz = np.min((1000, X.shape[0]))
        sigma_list = []
        for y in [Y[i][0] for i in range(len(Y))]:
            sigma_list.append(kernelwidthPair(X[0:siz], y[0:siz]))
        sigma = np.mean(sigma_list)

    
    Kxx = grbf(X,X,sigma)
    Kyy_all = []
    Kxy_all = []
    for i in range(len(Y)):
        Kyy_i = []
        kxy_i = []
        for j in range(len(Y[0])):
            Kyy_i.append(grbf(Y[i][j], Y[i][j], sigma))
            kxy_i.append(grbf(X, Y[i][j], sigma))
        Kyy_all.append(Kyy_i)
        Kxy_all.append(kxy_i)
    
    return Kxx, Kyy_all, Kxy_all
    
    
def np_diff_mmd_test(X, X2, Y, Y2, correction, alpha, beta = 0.7, margin = 8, kernel = 'rbf'):
    '''
    MMD based multi-sample hypothesis testing
    
    Args
    ----
    - X: the first test dataset
    - X2: the second test dataset
    - Y: the first fake dataset
    - Y2: the second fake dataset
    - correction: the method of multiple testing correction (see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
    - alpha: significance level
    - beta: decay rate
    - margin: the minimum proportion of arms to keep and to drop at each round
    - kernel: kernel function for MMD
    
    Output: the indices of the arms to keep to the next round
    
    '''
    
    dim = float(X.shape[1])
    mmd2_diffs = []
    test_stats = []
    # indicator of observing at least one significant result
    h = len(Y[0])
    coef = np.array([beta**i for i in range(h)])
    coef = np.concatenate((coef, -coef))
    
    mmd_mavg_all = []
    if len(Y) == 2:
        Y_concat = [[np.concatenate((Y[j][i],Y2[j][i]), axis = 0) for i in range(len(Y[0]))] for j in range(2)]
        Kxx, Kyy, Kxy = Kernels_multi(X, Y_concat)
        m = Kxx.shape[0]
    else:
        Kxx, Kyy, Kxy = Kernels_multi(X, Y)
        m = Kxx.shape[0]

    
    # compute moving average mmd
    for i in range(len(Y)):
        mmd_mavg = 0
        for j in range(len(Y[0])):               
            Kyynd = Kyy[i][j]-np.diag(np.diagonal(Kyy[i][j]))
            u_yy=np.sum(Kyynd)*( 1./(m*(m-1)) )
            u_xy=np.sum(Kxy[i][j])/(m*m)
            mmd = u_yy-2.*u_xy            
            mmd_mavg += mmd * beta**j

        mmd_mavg_all.append(mmd_mavg)
        
    sort_loss_idx = np.argsort(mmd_mavg_all)
    if len(Y) == 2:
        return sort_loss_idx
    
    p_value_all = []
    covar = np.zeros((2*h, 2*h))    
    
    Kxx, Kyy, Kxy = Kernels_multi(X2, Y2)
    # recompute mmd
    for i in range(len(Y2)):
        mmd_mavg = 0
        for j in range(len(Y2[0])):               
            Kyynd = Kyy[i][j]-np.diag(np.diagonal(Kyy[i][j]))
#             m = Kyy[i][j].shape[0]
            u_yy=np.sum(Kyynd)*( 1./(m*(m-1)) )
            u_xy=np.sum(Kxy[i][j])/(m*m)
            mmd = u_yy-2.*u_xy            
            mmd_mavg += mmd * beta**j

        mmd_mavg_all.append(mmd_mavg)
        
    # compute covaraince matrix
    for j in range(h):
#         covar[j][j] = MMD_Var(Kxx, Kyy[sort_loss_idx[0]][j], Kxy[sort_loss_idx[0]][j])
        covar[j][j] = _np_get_var(Kxx, Kyy[sort_loss_idx[0]][j], Kxy[sort_loss_idx[0]][j])
        for k in range(j):
#             covar[j][k] = MMD_coVar(Kxx, Kyy[sort_loss_idx[0]][j], Kyy[sort_loss_idx[0]][k], Kxy[sort_loss_idx[0]][j], Kxy[sort_loss_idx[0]][k]) 
            covar[j][k] = _np_get_covar(Kxx, Kyy[sort_loss_idx[0]][j], Kyy[sort_loss_idx[0]][k], Kxy[sort_loss_idx[0]][j], Kxy[sort_loss_idx[0]][k])
            
    for i in range(1, len(Y)):
        for j in range(h, 2*h):
#             covar[j][j] = MMD_Var(Kxx, Kyy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[i]][j-h])
            covar[j][j] = _np_get_var(Kxx, Kyy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[i]][j-h])
            for k in range(h):
#                 covar[j,k] = MMD_coVar(Kxx, Kyy[sort_loss_idx[i]][j-h], Kyy[sort_loss_idx[0]][k], Kxy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[0]][k])
                covar[j,k] = _np_get_covar(Kxx, Kyy[sort_loss_idx[i]][j-h], Kyy[sort_loss_idx[0]][k], Kxy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[0]][k])
            
            for k in range(h, j):
#                 covar[j,k] = MMD_coVar(Kxx, Kyy[sort_loss_idx[i]][j-h], Kyy[sort_loss_idx[i]][k-h], Kxy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[i]][k-h])
                covar[j,k] = _np_get_covar(Kxx, Kyy[sort_loss_idx[i]][j-h], Kyy[sort_loss_idx[i]][k-h], Kxy[sort_loss_idx[i]][j-h], Kxy[sort_loss_idx[i]][k-h])
        
        covar_m = covar.T + covar
        np.fill_diagonal(covar_m, np.diag(covar))
        
        var_est = coef.dot(covar_m).dot(coef.T)
        mmd2_diff = mmd_mavg_all[sort_loss_idx[0]] - mmd_mavg_all[sort_loss_idx[i]]
        ratio = mmd2_diff / np.sqrt(max(var_est, _eps))
        p_value = scipy.stats.norm.cdf(ratio)    
        p_value_all.append(p_value)
    
    
    test_resuls, adjuested_p = statsmodels.stats.multitest.multipletests(p_value_all, alpha = alpha, method = correction)[:2]
    insig_index = np.where(test_resuls == False)[0]

    min_n = max(2, np.ceil((len(test_resuls)+1)/margin))
    max_n = np.floor((len(test_resuls)+1)*(margin-1)/margin)

    if len(insig_index)+1 < min_n:
        insig_index = np.argpartition(-adjuested_p, int(min_n-1))[:int(min_n-1)]
    elif len(insig_index)+1 > max_n:
        insig_index = np.argpartition(-adjuested_p, int(max_n-1))[:int(max_n-1)]
    
    result = []
    result.append(sort_loss_idx[0])
    for e in insig_index:
        result.append(sort_loss_idx[e+1])
    return result
        

    

def _np_get_var(K_XX, K_YY, K_XY):
    m = float(K_YY.shape[0])  # Assumes X, Y, Z are same shape
    diag_X = np.diag(K_XX)
    diag_Y = np.diag(K_YY)
    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    
    sum_diag2_X = np.dot(diag_X, diag_X)
    sum_diag2_Y = np.dot(diag_Y, diag_Y)
    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    
    E_x_muX_sq = (np.dot(Kt_XX_sums, Kt_XX_sums) - Kt_XX_2_sum) / (m*(m-1)*(m-2))
    E_y_muY_sq = (np.dot(Kt_YY_sums, Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    
    
    K_XY_sums_1 = K_XY.sum(axis=1)
    K_XY_2_sum  = (K_XY ** 2).sum()
    E_x_muY_sq = (np.dot(K_XY_sums_1, K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    
    K_XY_sums_0 = K_XY.sum(axis=0)
    E_y_muX_sq = (np.dot(K_XY_sums_0, K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    E_y_muY_y_muX = np.dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_x_muX_x_muY = np.dot(Kt_XX_sums, K_XY_sums_1) / (m*m*(m-1))
    
    E_kxx2 = Kt_XX_2_sum / (m * (m-1))
    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kxy2 = K_XY_2_sum / (m * m)
    
    muX_muX = Kt_XX_sums.sum() / (m * (m-1))
    muY_muY = Kt_YY_sums.sum() / (m * (m-1))
    muX_muY = K_XY_sums_0.sum() / (m * m)
    
    first_order = 4 * (m-2) / (m * (m-1)) * (
          E_x_muX_sq - muX_muX**2 # ADDED
        + E_y_muY_sq - muY_muY**2
        
        + E_x_muY_sq - muX_muY**2
        + E_y_muX_sq - muX_muY**2
        
        - 2 * E_y_muY_y_muX + 2 * muY_muY * muX_muY
        - 2 * E_x_muX_x_muY + 2 * muX_muX * muX_muY # ADDED
    )
        
    second_order = 2 / (m * (m-1)) * (
          E_kxx2 - muX_muX**2 # ADDED
        + E_kyy2 - muY_muY**2
        + 2 * E_kxy2 - 2 * muX_muY**2
        - 4 * E_y_muY_y_muX + 4 * muY_muY * muX_muY
        - 4 * E_x_muX_x_muY + 4 * muX_muX * muX_muY # ADDED
    )    
    
    return first_order + second_order
        
        



def _np_get_covar(K_XX, K_YY, K_ZZ, K_XY, K_XZ):
    m = float(K_YY.shape[0])  # Assumes X, Y, Z are same shape
    diag_X = np.diag(K_XX)
    diag_Y = np.diag(K_YY)
    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    
    sum_diag2_X = np.dot(diag_X, diag_X)
    sum_diag2_Y = np.dot(diag_Y, diag_Y)
    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y
    
    E_x_muX_sq = (np.dot(Kt_XX_sums, Kt_XX_sums) - Kt_XX_2_sum) / (m*(m-1)*(m-2))
    E_y_muY_sq = (np.dot(Kt_YY_sums, Kt_YY_sums) - Kt_YY_2_sum) / (m*(m-1)*(m-2))
    
    
    K_XY_sums_1 = K_XY.sum(axis=1)
    K_XZ_sums_1 = K_XZ.sum(axis=1)
    K_XY_2_sum  = (K_XY ** 2).sum()
    E_x_muY_sq = (np.dot(K_XY_sums_1, K_XY_sums_1) - K_XY_2_sum) / (m*m*(m-1))
    
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XZ_sums_0 = K_XZ.sum(axis=0)
    E_y_muX_sq = (np.dot(K_XY_sums_0, K_XY_sums_0) - K_XY_2_sum) / (m*m*(m-1))
    
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    E_y_muY_y_muX = np.dot(Kt_YY_sums, K_XY_sums_0) / (m*m*(m-1))
    E_x_muX_x_muY = np.dot(Kt_XX_sums, K_XY_sums_1) / (m*m*(m-1))
    E_x_muX_x_muZ = np.dot(Kt_XX_sums, K_XZ_sums_1) / (m*m*(m-1))
    
    E_x_muY_x_muZ = np.dot(K_XY_sums_1, K_XZ_sums_1) / (m*m*m)
    
    E_kxx2 = Kt_XX_2_sum / (m * (m-1))
    E_kyy2 = Kt_YY_2_sum / (m * (m-1))
    E_kxy2 = K_XY_2_sum / (m * m)
    
    muX_muX = Kt_XX_sums.sum() / (m * (m-1))
    muY_muY = Kt_YY_sums.sum() / (m * (m-1))
    muX_muY = K_XY_sums_0.sum() / (m * m)
    muX_muZ = K_XZ_sums_0.sum() / (m * m)
    
    first_order = 4 * (m-2) / (m * (m-1)) * (
          E_x_muX_sq - muX_muX**2 # ADDED       
        - E_x_muX_x_muY + muX_muX * muX_muY # ADDED
        - E_x_muX_x_muZ + muX_muX * muX_muZ # ADDED
        + E_x_muY_x_muZ - muX_muY * muX_muZ
    )
    

    second_order = 2 / (m * (m-1)) * (
          E_kxx2 - muX_muX**2 # ADDED
        - 2 * E_x_muX_x_muY + 2 * muX_muX * muX_muY # ADDED        
        - 2 * E_x_muX_x_muZ + 2 * muX_muX * muX_muZ # ADDED
        + 2 * E_x_muY_x_muZ - 2 * muX_muY * muX_muZ
    ) 
    
    return first_order + second_order

