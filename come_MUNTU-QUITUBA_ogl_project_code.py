import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path

EXCEL_PATH = Path(__file__).resolve().with_name("dataset.csv")

def load(path):
    data = pd.read_csv(path)
    return data

def fit_surrogates(data, seed=42):
    X = data[["S","l","e","vrf","T","Sc"]].values
    y = data["L"].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)

    ols = LinearRegression().fit(X_tr, y_tr)
    pred = ols.predict(X_te)
    r2 = r2_score(y_te, pred)
    rmse = math.sqrt(mean_squared_error(y_te, pred))

    # Physics L = k * S / (l*vrf + e)
    S = X_tr[:,0]; l = X_tr[:,1]; e = X_tr[:,2]; vrf = X_tr[:,3]
    phi = S / (l*vrf + e)
    k = float(phi.dot(y_tr) / phi.dot(phi))

    S = X_te[:,0]; l = X_te[:,1]; e = X_te[:,2]; vrf = X_te[:,3]
    phi_te = S / (l*vrf + e)
    pred_phys = k * phi_te
    r2_phys = r2_score(y_te, pred_phys)
    rmse_phys = math.sqrt(mean_squared_error(y_te, pred_phys))

    resid = y_tr - ols.predict(X_tr)
    sigma2 = float(np.var(resid, ddof=1))
    return ols, k, (r2, rmse, r2_phys, rmse_phys, sigma2)

def solve_variance_tightening(ols, data, sigma2, target_factor=2.0, rmin=0.1): # target_factor=2 means halve std 
    y = data["L"].values
    var_current = float(np.var(y, ddof=1))
    var_target = var_current / (target_factor**2)

    Xvars = data[["S","l","e","vrf","T","Sc"]].var(ddof=1).values
    beta = ols.coef_
    a = (beta**2) * Xvars  # contributions

    b = var_target - sigma2  # controllable budget
    if b <= 0:
        raise ValueError("Target variance is below irreducible residual variance; infeasible.")

    # Representative costs 
    c = np.array([2.0, 1.5, 1.0, 5.0, 0.1, 1.0], dtype=float)

    def r_of_lam(lam):
        r = c / (c + lam*a)
        return np.clip(r, rmin, 1.0)

    def f(lam):
        r = r_of_lam(lam)
        return float(np.sum(a*r**2) - b)

    # Bisection on lambda
    lo, hi = 0.0, 1.0
    while f(hi) > 0:
        hi *= 2
        if hi > 1e18:
            raise ValueError("Infeasible with rmin bound; cannot reach target variance.")

    for _ in range(200):
        mid = 0.5*(lo+hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid

    lam = hi
    r = r_of_lam(lam)
    obj = float(np.sum(c*(1-r)**2))
    return r, obj, var_current, var_target, a, c

def primal_dual_gne(a, c, sigma2, var_target, rmin=0.1, iters=20000, alpha=0.02, beta_dual=0.5):
    # Scale the constraint to O(1) for stable stepsizes
    b = var_target - sigma2
    a_scaled = a / b

    r = np.ones_like(c)
    lam = 0.0
    hist = []
    for t in range(iters):
        grad_cost = 2*c*(r-1)
        grad = grad_cost + 2*lam*a_scaled*r
        r = np.clip(r - alpha*grad, rmin, 1.0)
        lam = max(0.0, lam + beta_dual*(np.sum(a_scaled*r**2) - 1.0))
        if t % 10 == 0:
            hist.append((float(np.sum(c*(1-r)**2)), float(np.sum(a_scaled*r**2)-1.0), lam))
    return r, lam, np.array(hist)

def main():
    data = load(EXCEL_PATH)
    ols, k, metrics = fit_surrogates(data)
    r2, rmse, r2_phys, rmse_phys, sigma2 = metrics

    print("OLS:    R2=%.4f RMSE=%.6f" % (r2, rmse))
    print("Phys:   R2=%.4f RMSE=%.6f (k=%.6f)" % (r2_phys, rmse_phys, k))
    print("sigma (resid) = %.6f" % math.sqrt(sigma2))

    r_star, obj, var_current, var_target, a, c = solve_variance_tightening(ols, data, sigma2)
    print("Current Var(L)=%.3e  Target Var(L)=%.3e" % (var_current, var_target))
    print("Optimal r =", r_star)
    print("Objective =", obj)

    #Figure: 
    names = ["S","l","e","vrf","T","Sc"]
    order = np.argsort(-a)
    plt.figure()
    plt.bar(np.array(names)[order], a[order])
    plt.ylabel("Contribution to Var(L) (approx.)")
    plt.title("Variance contributions via linear surrogate")
    plt.tight_layout()
    plt.savefig("fig_contrib.png", dpi=200)
    plt.close()

    #Primal-dual convergence
    r_pd, lam_pd, hist = primal_dual_gne(a, c, sigma2, var_target)
    plt.figure()
    plt.plot(hist[:,0])
    plt.xlabel("Iteration (x10)")
    plt.ylabel("Objective")
    plt.title("Primal-dual dynamics (objective)")
    plt.tight_layout()
    plt.savefig("fig_convergence.png", dpi=200)
    plt.close()

    print("Primal-dual r =", r_pd)

if __name__ == "__main__":
    main()
