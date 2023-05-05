import numpy as np
import sklearn.metrics
# import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge, LinearRegression

# model for comparison
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# import mean_squared_error
from sklearn.metrics import mean_squared_error


""" utility functions """

def save_figure(fig: object, name: str, path: str):

    """
    save a figure in a path

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        figure to save
    name : str
        name of the file
    path : str
        path where to save the figure
    """


    fig.savefig(path + '/' + name + '.png', dpi=300, bbox_inches='tight')

    print("Figure saved in", path + '/' + name + '.png')


""" DATA """

def FrankeFunction(x: np.ndarray, y: np.ndarray, noise_scale=0.1) -> np.ndarray:
    
    """
    return the the Franke Function
    
    Parameters
    ---------- x : np.ndarray
    y : np.ndarray
    noise_scale : float
        scale of the noise term, default 0.1
        
    Returns:
    np.ndarray : 
        application of the function f(x, y)
    """
    
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    
    noise_term = np.random.normal(0, noise_scale, x.shape)
    
    
    return term1 + term2 + term3 + term4 + noise_term



def generate_data(N: int, random_ax: bool=True, noise_scale: float=0.1, scaling: bool=False, mean_center=False) -> tuple:
    
    """
    generate the data for training and 2d mesh
    
    Parameters
    ----------
    N : int
        size
    random_ax : bool
        if True x, y will be drawn from rand() else arange, default True
    noise_scale : float
        standard devation of the noise term drawn from a Normal with zero mean, default 0.1
    scaling : bool
        if True, scale the data, default False
    mean_center : bool
        if True, mean center the data, default False
        
    Returns
    -------
    np.ndarray : x
        (N, 1)
    np.ndarray : y
        (N, 1)
    np.ndarray : z
        (N, 1)
        
    np.ndarray : xm
        (N, N)
    np.ndarray : ym
        (N, N)
    np.ndarray : zm
        (N, N)
    """
    
    if random_ax:
        x = np.random.rand(N, 1)
        y = np.random.rand(N, 1)
    else:
        x = np.linspace(0, 1, N).reshape(N, 1)
        y = np.linspace(0, 1, N).reshape(N, 1)

    # define the target
    z = FrankeFunction(x, y, noise_scale=noise_scale)


    # make meshgrid
    xm, ym = np.meshgrid(x,y)
    zm = FrankeFunction(xm, ym, noise_scale=noise_scale)

    # scale the data
    if scaling:
        x, y, xm, ym = (preprocessing.StandardScaler().fit_transform(x), 
                        preprocessing.StandardScaler().fit_transform(y), 
                        preprocessing.StandardScaler().fit_transform(xm), 
                        preprocessing.StandardScaler().fit_transform(ym))

    # mean center the data
    if mean_center:
        x, y, xm, ym = (mean_centering(x),
                        mean_centering(y),
                        mean_centering(xm),
                        mean_centering(ym))
   
    return (x, y, z), (xm, ym, zm)


def polynomial_data(degree: int, N: int, true_beta=None, random_ax=True, noise_scale=0, verbose=False) -> tuple:

    """
    generate the data from a polynomial of degree=degree

    Parameters
    ----------
    degree : int
        degree of the polynomial
    N : int
        size
    true_beta : list [optional]
        list of the coefficients of the polynomial, default None
    random_ax : bool
        if True, x will be drawn from rand() else arange, default True
    noise_scale : float
        standard devation of the noise term drawn from a Normal with zero mean, default 0
    verbose : bool
        if True, print the polynomial, default False

    Returns
    -------
    np.ndarray : x
    np.ndarray : z
    """

    if random_ax:
        x = np.random.uniform(0, 10, size=(N, 1))
    else:
        x = np.linspace(0, 10, N).reshape(N, 1)

    # create random coefficients
    if true_beta is None:
        true_beta = np.random.normal(0, 0.01, size=(degree + 1, 1))
    else:
        true_beta = np.array(true_beta).reshape(-1, 1)

    # check degree and priorize the provided beta
    if degree != len(true_beta) - 1:
        degree = len(true_beta) - 1

    # define design matrix 
    D = build_design_matrix(X=x, degree=degree)

    # add noise 
    noise = np.random.normal(0, noise_scale, size=(N, 1))

    # print 
    if verbose:
        print("true polynomial : z = ", end="")
        for i in range(len(true_beta)):
            if i != 0: print(" + ", end="")
            print(f"({true_beta[i, 0]:.4f})*x^{i}", end="")
        print()

    return x, D @ true_beta + noise, true_beta


def mean_centering(X: np.ndarray) -> np.ndarray:

    """
    mean centering of the data

    Parameters
    ----------
    X : np.ndarray
        (N, M)

    Returns
    -------
    np.ndarray : X
        (N, M)
    """

    return X - np.mean(X, axis=0)


""" DESIGN MATRIX """

def get_coefficients(nb_vars: int, degree: int, verbose=False) -> list:

    """
    return the coefficients of a polynomial of degree=degree and $nb_vars dimensions 
    in the form of a list

    Parameters
    ----------
    nb_vars : int
        number of variables
    degree : int
        degree of the polynomial
    verbose : bool
        if True, the polynomial is printed, default False

    Returns
    -------
    list : coefficients
    """

    # check 
    assert degree > 0, "degree must be greater that zero"
    assert 0 < nb_vars < 3, "the number of variables must be between 1 and 2"

    # one dimension case
    if nb_vars == 1:
        return [i for i in range(1, degree + 1)]

    # two dimensions case
    def combinations(n: int) -> list:

        """
        return the combinations of the variables' degrees of the form (i, n-i) where i = 0, 1, ..., n
        """
        combs = []
        for i in range(n + 1):
            combs += [i]
            combs += [n - i]
        return combs

    coeff = []
    for d in range(1, degree + 1):
        coeff += combinations(d)
        
    if verbose:
        func = []
        for i in range(0, len(coeff), 2):
            func.append(f"x^{coeff[i]}*y^{coeff[i + 1]}")

        print(f'(len: {len(func)}) | form z =', end='')
        for j, pair in enumerate(func):
            if j != 0: print("+", end='')
            print(f" {pair}", end=" ")
    
    return coeff


def build_design_matrix(X: np.ndarray, Y=None, degree=1, intercept=False) -> np.ndarray:

    """
    return the design matrix of an input with one or two features, and degree=degree

    Parameters
    ----------
    X : np.ndarray
        (n, 1) feature 1 
    Y : np.ndarray [optional]
        (n, 1) feature 2, default None
    degree : int
        degree of the polynomial, default 1
    intercept : bool
        if True, the intercept is added, default False

    Returns
    -------
    np.ndarray : design matrix
    """
    
    # size of the data
    n = len(X)
    trg_ax = 1
    D = np.ones((n, 1)) if intercept else np.zeros((n, 1))
    
    # one dimension case
    if Y is None:
        coeff = get_coefficients(nb_vars=1, degree=degree)
        for i in range(len(coeff)):
            D = np.concatenate([D, X**coeff[i]], axis=trg_ax)
        return D

    # two dimensions case
    # mesh data case
    if X.shape[1] > 1:
        X = X.reshape(n, n, 1)
        Y = Y.reshape(n, n, 1)
        trg_ax = 2
        D = np.ones((n, n, 1)) if intercept else np.zeros((n, n, 1))
    
    coeff = get_coefficients(nb_vars=2, degree=degree)
    
    # build the full Design matrix
    for i in range(0, len(coeff), 2):
        
        D = np.concatenate([D, X**coeff[i] * Y**coeff[i+1]], axis=trg_ax)
        
    return D


def build_design_matrix_2(X: np.ndarray, Y=None, degree=1, intercept=False) -> np.ndarray:

    """
    return the design matrix of an input with one or two features, and degree=degree

    Parameters
    ----------
    X : np.ndarray
        (n, 1) feature 1 
    Y : np.ndarray [optional]
        (n, 1) feature 2, default None
    degree : int
        degree of the polynomial, default 1
    intercept : bool
        if True, the intercept is added, default False

    Returns
    -------
    np.ndarray : design matrix
    """

    # size of the data
    n = len(X)
    trg_ax = 1
    D = np.ones((n, 1)) 

    # one dimension case
    if Y is None:
        for i in range(1, degree + 1):
            D = np.concatenate([D, X**i], axis=trg_ax)
        return D

    # two dimensions case
    # mesh data case
    if X.shape[1] > 1:
        X = X.reshape(n, n, 1)
        Y = Y.reshape(n, n, 1)
        trg_ax = 2
        D = np.ones((n, n, 1))

    # build the full Design matrix
    for i in range(1, degree + 1):
        for j in range(i + 1):
            D = np.concatenate([D, X**(i - j) * Y**j], axis=trg_ax)

    return D


""" EVALUATION METRICS """

def CoD(Z_true: np.ndarray, Z_pred: np.ndarray) -> float:

    """
    return the Coefficient of Determination

    Parameters
    ----------
    Z_true : np.ndarray
        the true values
    Z_pred : np.ndarray
        the predicted values

    Returns
    -------
    float : CoD
    """
    
    assert Z_true.shape == Z_pred.shape, f"!shape mismatch : {Z_true.shape=} != {Z_pred.shape=}"
    
    rss = ((Z_true - Z_pred)**2).sum()
    tss = ((Z_true - Z_true.mean())**2).sum()
    
    return 1 - rss / tss


def MSE(Z_true: np.ndarray, Z_pred: np.ndarray) -> float:

    """
    return the Mean Squared Error

    Parameters
    ----------
    Z_true : np.ndarray
        the true values
    Z_pred : np.ndarray
        the predicted values

    Returns
    -------
    float : MSE
    """
    
    assert Z_true.shape == Z_pred.shape, f"!shape mismatch : {Z_true.shape=} != {Z_pred.shape=}"
    
    # return ((Z_true - Z_pred)**2).mean()
    return mean_squared_error(Z_true, Z_pred)


""" RESAMPLING METHODS """

# K-FOLD

def manual_folding(dataset_x: np.ndarray, dataset_z: np.ndarray, K: int) -> list:
    
    """
    implement a k-fold data splitting
    
    Parameters
    ----------
    dataset_x : np.ndarray
        input
    dataset_z : np.ndarray
        target
    K : int 
        number of folds
        
    Returns
    -------
    list : [[@ + + ... +],
            [+ @ + ... +],
            [+ + @ ... +],
            ...
            [+ + + ... @]]
            
            + : training fold
            @ : test fold
            
        dataset folded each time moving the test fold rightward
    list : same but for the test set
    """
    
    # numbers
    size = dataset_x.shape[0]
    ndim_x = dataset_x.shape[1]
    ndim_z = dataset_z.shape[1]
    
    index_range = range(size)
    
    # size of each fold
    fold_size = size // K
    
    list_of_folds_x = []
    list_of_folds_z = []
    
    # loop over folds
    for k in range(K):
        
        # define indexes of the elements for the test fold
        test_indexes = list(range(k*fold_size, k*fold_size + fold_size))
        
        # defined indexes of train fold by removing test fold indexes
        train_indexes = [i for i in index_range if i not in test_indexes]
        
        # append the actual elements
        list_of_folds_x += [[dataset_x[train_indexes].reshape(-1, ndim_x), 
                             dataset_x[test_indexes].reshape(-1, ndim_x)]]
        list_of_folds_z += [[dataset_z[train_indexes].reshape(-1, ndim_z), 
                             dataset_z[test_indexes].reshape(-1, ndim_z)]]
        
    return list_of_folds_x, list_of_folds_z


def folding_from_sklearn(dataset_x: np.ndarray, dataset_z: np.ndarray, K: int) -> list:

    """
    implement a k-fold data splitting using sklearn

    Parameters
    ----------
    dataset_x : np.ndarray
        design matrix
    dataset_z : np.ndarray
        vector of targets
    K : int
        number of folds

    Returns
    -------
    list : [[@ + + ... +],
            [+ @ + ... +],
            [+ + @ ... +],
            ...
            [+ + + ... @]]

            + : training fold
            @ : test fold

        dataset folded each time moving the test fold rightward
    list : same but for the test set
    """

    # KFold method from sklearn
    kf = KFold(n_splits=K)

    # list of folds
    list_of_folds_x = []
    list_of_folds_z = []

    # loop folds iterations
    for train_index, test_index in kf.split(dataset_x):

        # append
        list_of_folds_x += [[dataset_x[train_index], dataset_x[test_index]]]
        list_of_folds_z += [[dataset_z[train_index], dataset_z[test_index]]]

    return list_of_folds_x, list_of_folds_z


def folding_tensor_from_sklearn(dataset_x: np.ndarray, dataset_z: np.ndarray, K: int) -> (list, list):
    """
        implement a k-fold tensor splitting using sklearn

        Parameters
        ----------
        dataset_x : tf.tensor
            design matrix
        dataset_z : tf.tensor
            vector of targets
        K : int
            number of folds

        Returns
        -------
        list : [[@ + + ... +],
                [+ @ + ... +],
                [+ + @ ... +],
                ...
                [+ + + ... @]]

                + : training fold
                @ : test fold

            dataset folded each time moving the test fold rightward
        list : same but for the test set
        """


    # KFold method from sklearn
    kf = KFold(n_splits=K)

    # list of folds
    list_of_folds_x = []
    list_of_folds_z = []

    # loop folds iterations
    for train_index, test_index in kf.split(dataset_x):

        # append
        list_of_folds_x += [[tf.gather(dataset_x, train_index), tf.gather(dataset_x, test_index)]]
        list_of_folds_z += [[tf.gather(dataset_z, train_index), tf.gather(dataset_z, test_index)]]

    return list_of_folds_x, list_of_folds_z


def cross_validation(X: np.ndarray, Z: np.ndarray, K: int, source: str) -> list:

    """
    K-fold cross validation

    Parameters
    ----------
    X : np.ndarray
        input data
    Z : np.ndarray
        target data
    K : int
        number of folds
    source : str
        "manual" or "sklearn"

    Returns
    -------
    list : [[@ + + ... +],
            [+ @ + ... +],
            [+ + @ ... +],
            ...
            [+ + + ... @]]

            + : training fold
            @ : test fold

        dataset folded each time moving the test fold rightward
    list : same but for the test set
    """

    if source == "manual":
        return manual_folding(X, Z, K)

    elif source == "sklearn":
        return folding_from_sklearn(X, Z, K)

    else:
        raise ValueError("source must be 'manual' or 'sklearn'")


""" MODEL SELECTION """

# our own implementation of the OLS method
def rOLS(dataset_x: list, dataset_z: list, ridge=False, lasso=False, lambda_r=None) -> tuple:

    """
    Ordinary Least Squares and Ridge Regression [optional] or Lasso Regression [optional]

    Parameters
    ----------
    dataset_x : list
        list of np.ndarray
    dataset_z : list
        list of np.ndarray
    ridge : bool
        if True, ridge regression is performed, default False
    lasso : bool
        if True, lasso regression is performed, default False
    lambda_r : float
        regularization parameter, default None

    Returns
    -------
    np.ndarray : beta
    list : MSE scores 
    list : CoD scores
    np.ndarray : training and test predictions
    """

    # define data 
    X_train, X_test = dataset_x
    Z_train, Z_test = dataset_z

    # if ridge regression is not selected, set the parameter to zero
    if not (ridge or lasso):
        lambda_r = 0

    ###### TRAINING ######

    if lasso:
        # do lasso regression with scikit-learn
        lasso_model = Lasso(alpha=lambda_r)
        lasso_model.fit(X_train, Z_train)
        beta = np.hstack((lasso_model.intercept_, lasso_model.coef_))

        # training predictions
        Z_pred_train = lasso_model.predict(X_train).reshape(-1, 1)

        # test predictions
        Z_pred_test = lasso_model.predict(X_test).reshape(-1, 1)
    else:
        ridge_reg = lambda_r * np.eye(X_train.shape[1]) 
        assert ridge_reg.shape == (X_train.shape[1], X_train.shape[1]), f"!shape mismatch : {ridge_reg.shape=} != {X_train.shape[1], X_train.shape[1]=}"

        # compute optimal beta with matrix inversion
        beta = np.linalg.pinv(X_train.T @ X_train + ridge_reg) @ X_train.T @ Z_train
    

        # training predictions
        Z_pred_train = (X_train @ beta).reshape(-1, 1)

        # test predictions
        Z_pred_test = (X_test @ beta).reshape(-1, 1)

    ###### EVALUATION ######

    # evaluation training
    mse_train = MSE(Z_true=Z_train, Z_pred=Z_pred_train)
    cod_train = CoD(Z_true=Z_train, Z_pred=Z_pred_train)

    # evaluation test
    mse_test = MSE(Z_true=Z_test, Z_pred=Z_pred_test)
    cod_test = CoD(Z_true=Z_test, Z_pred=Z_pred_test)

    return beta, [mse_train, mse_test], [cod_train, cod_test], Z_pred_test

# SciKit-Learn implementation of the OLS method
def rOLS_sklearn(dataset_x: list, dataset_z: list, degree: int, intercept=False, ridge=False, lasso=False, lambda_r=None) -> tuple:

    """
    Ordinary Least Squares and Ridge Regression [optional] or Lasso Regression [optional]

    Parameters
    ----------
    dataset_x : list
        list of np.ndarray
    dataset_z : list
        list of np.ndarray
    degree : int
        degree of the polynomial
    intercept : bool
        if True, the model is fitted with an intercept, default False
    ridge : bool
        if True, ridge regression is performed, default False
    lasso : bool
        if True, lasso regression is performed, default False
    lambda_r : float
        regularization parameter, default None

    Returns
    -------
    np.ndarray : beta
    list : MSE scores 
    list : CoD scores
    np.ndarray : training and test predictions
    """

    # define data 
    X_train, X_test = dataset_x
    Z_train, Z_test = dataset_z

    ###### TRAINING ######

    if lasso:
        # do lasso regression with scikit-learn
        beta = Lasso(alpha=lambda_r).fit(X_train, Z_train).coef_
    elif ridge:
        # do ridge regression with scikit-learn
        beta = Ridge(alpha=lambda_r).fit(X_train, Z_train).coef_
    else:
        # do OLS with scikit-learn
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression(fit_intercept=intercept))
        model.fit(X_train, Z_train)
        beta = model.steps[1][1].coef_.reshape(-1, 1)


    # training predictions
    Z_pred_train = model.predict(X_train).reshape(-1, 1)

    # evaluation
    mse_train = MSE(Z_true=Z_train, Z_pred=Z_pred_train)
    cod_train = CoD(Z_true=Z_train, Z_pred=Z_pred_train)

    ###### TEST ######

    # test predictions
    Z_pred_test = model.predict(X_test).reshape(-1, 1)

    # evaluation
    mse_test = MSE(Z_true=Z_test, Z_pred=Z_pred_test)
    cod_test = CoD(Z_true=Z_test, Z_pred=Z_pred_test)

    return beta, [mse_train, mse_test], [cod_train, cod_test], Z_pred_test




def logistic_function(x, betas):
    """
    Logistic function
    Parameters
    ----------
    x : tf.tensor
        input data
    betas : tf.tensor
        logistic regression parameters

    Returns
    -------
    tf.tensor : logistic function
    """
    feature_input = x*betas[1:]
    exp_term = -(betas[0] + tf.reduce_sum(feature_input, axis=1))
    return 1 / (1+tf.math.exp(exp_term))


def cross_entropy(y_true, prediction):
    """
    Cross entropy loss function
    Parameters
    ----------
    y_true: tf.tensor
        ground truth labels
    prediction: tf.tensor
        predicted probabilities

    Returns
    -------
    tf.tensor : cross entropy loss
    """
    ce_eps = 1e-7
    prediction = tf.clip_by_value(prediction, ce_eps, 1-ce_eps) #clip to avoid numerical problems with log(0). In general, logit= 0 or 1 shouldn't be possible, but these values come up due to the limited float32 precision
    return -tf.reduce_mean(y_true * tf.math.log(prediction) + (1-y_true)*tf.math.log(1-prediction))
    


