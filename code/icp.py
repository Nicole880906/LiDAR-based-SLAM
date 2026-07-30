from scipy.spatial import KDTree

import numpy as np

def kabsch_2D(z, m):
    mean_z = np.mean(z, axis=0)
    mean_m = np.mean(m, axis=0)
    delta_z = z - mean_z
    delta_m = m - mean_m

    Q = delta_z.T @ delta_m
    U, S, Vt = np.linalg.svd(Q)

    R = Vt.T @ np.diag([1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    p = mean_m - R @ mean_z

    return R, p

def icp(source, target, T_org ,iter=50, tolerance=1e-5):
    trans_mat = T_org
    pre_err = float('inf')
    target_tree = KDTree(target)

    new_source = (trans_mat[:2, :2] @ source.T).T + trans_mat[:2, 2]
    
    for i in range(iter):
        distance, idx = target_tree.query(new_source, k = 1)
        err = np.mean(distance)
        
        if abs(pre_err - err) < tolerance:
            break
        
        pre_err = err
        matched_target = target[idx]
        R, p = kabsch_2D(new_source, matched_target)
        new_source = (R @ new_source.T).T + p
        
        T_step = np.eye(3)
        T_step[:2, :2] = R
        T_step[:2, 2] = p
        trans_mat = T_step @ trans_mat

    return trans_mat, pre_err
