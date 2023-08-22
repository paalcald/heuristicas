import numpy as np
from prsbc.utilidades.constantes import *
from prsbc.heuristicas.hcv import hcv_fleet

def vnd_extended(r, y, b, tt, z, v, p, a, q, t, k, tl, D, N):
    zmin = z
    for d in np.arange(1, D):
        for idx, rx in np.ndenumerate(r[1:, 1:-1]):
            idx = (idx[0] + 1, idx[1] + 1)
            for idy, ry in np.ndenumerate(r[idx[0] + 1 :, 1:-1]):
                idy = (idy[0] + idx[0] + 1, idy[1] + 1)
                if ry in N[rx, :d]:
                    maybe_b = b[idx] - y[idx] + y[idy]
                    if (maybe_b > k[idx[0]] or maybe_b < 0):
                        break #no factible
                    ro = r.copy()
                    ro[idx] = ro[idy]
                    yo = y.copy()
                    yo[idx] = yo[idy]
                    ao = a.copy()
                    ao[r[idx]] += yo[idx]
                    bo = b.copy()
                    b[idx] = maybe_b
                    tto = tt.copy()
                    tto[idx] += (
                        +t[r[idx[0], idx[1] - 1], r[idy]]
                        - t[r[idx[0], idx[1] - 1], r[idx]]
                        )
                    ro, yo, bo, ao, tt, z = hcv_fleet(v,p,q,t,k,tl,idx,ro,yo,ao,bo,tt)  
                    if z < zmin:
                        zmin = z
                        r = ro
                        y = yo
                        b = bo
                        a = ao
                        tt = tto

    return r, y, b, a, tt, zmin

                    
def vnd_basic(r, y, b, tt, a, q, t, k, tl, D, N):
    impv = 0
    z = None
    for d in np.arange(1, D):
        idxo = None
        idyo = None
        for idx, rx in np.ndenumerate(r[1:, 1:-1]):
            idx = (idx[0] + 1, idx[1] + 1)
            for idy, ry in np.ndenumerate(r[idx[0] + 1 :, 1:-1]):
                idy = (idy[0] + idx[0] + 1, idy[1] + 1)
                if ry in N[rx, :d]:
                    maybe_impv = get_impv_for_swap(idx, idy, r, y, b, tt, t, k, tl)
                    if maybe_impv is not None:
                        if impv < maybe_impv:
                            impv = maybe_impv
                            idxo = idx
                            idyo = idy
        if idxo is not None and idyo is not None:
            r, y, b, tt, z = swap(idxo, idyo, r, y, b, tt, a, q, t)
    if z is None:
        z = (
            np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
            + W_2 * np.sum(np.sum(np.abs(y)))
            + W_3 * np.sum(tt[:, -2])
        )
    return r, y, b, tt, z


def swap(idx, idy, ro, yo, bo, tto, a, q, t):
    tt = tto.copy()
    tt[idx[0], idx[1]] += (
        +t[ro[idx[0], idx[1] - 1], ro[idy]]
        - t[ro[idx[0], idx[1] - 1], ro[idx]]
    )
    tt[idy[0], idy[1]] += (
        +t[ro[idy[0], idy[1] - 1], ro[idx]]
        - t[ro[idy[0], idy[1] - 1], ro[idy]]
    )
    tt[idx[0], idx[1] + 1:] += (
        +t[ro[idx[0], idx[1] - 1], ro[idy]]
        + t[ro[idy], ro[idx[0], idx[1] + 1]]
        - t[ro[idx[0], idx[1] - 1], ro[idx]]
        - t[ro[idx], ro[idx[0], idx[1] + 1]]
    )
    tt[idy[0], idy[1]:] += (
        +t[ro[idy[0], idy[1] - 1], ro[idx]]
        + t[ro[idx], ro[idy[0], idy[1] + 1]]
        - t[ro[idy[0], idy[1] - 1], ro[idy]]
        - t[ro[idy], ro[idy[0], idy[1] + 1]]
    )
    z = (
        np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
        + W_2 * np.sum(np.sum(np.abs(yo)))
        + W_3 * np.sum(tt[:, -2])
    )
    b = bo.copy()
    b[idx[0], idx[1] : -1] += yo[idy] - yo[idx]
    b[idy[0], idy[1] : -1] += yo[idy] - yo[idx]
    y = yo.copy()
    y[idx], y[idy] = y[idy], y[idx]
    r = ro.copy()
    r[idx], r[idy] = r[idy], r[idx]
    return r, y, b, tt, z


def get_impv_for_swap(idx, idy, ro, yo, bo, tto, t, k, tl):
    tt = tto.copy()
    zmin = tto[idx[0], -2] + tto[idy[0], -2]
    tt[idx[0], idx[1]] += (
        +t[ro[idx[0], idx[1] - 1], ro[idy]]
        - t[ro[idx[0], idx[1] - 1], ro[idx]]
    )
    tt[idy[0], idy[1]] += (
        +t[ro[idy[0], idy[1] - 1], ro[idx]]
        - t[ro[idy[0], idy[1] - 1], ro[idy]]
    )
    tt[idx[0], idx[1] + 1: - 1] += (
        +t[ro[idx[0], idx[1] - 1], ro[idy]]
        + t[ro[idy], ro[idx[0], idx[1] + 1]]
        - t[ro[idx[0], idx[1] - 1], ro[idx]]
        - t[ro[idx], ro[idx[0], idx[1] + 1]]
    )
    tt[idy[0], idy[1] + 1: -1] += (
        +t[ro[idy[0], idy[1] - 1], ro[idx]]
        + t[ro[idx], ro[idy[0], idy[1] + 1]]
        - t[ro[idy[0], idy[1] - 1], ro[idy]]
        - t[ro[idy], ro[idy[0], idy[1] + 1]]
    )
    z = tt[idx[0], -2] + tt[idy[0], -2]
    if tt[idx[0], -2] > tl[idx[0]] or tt[idy[0], -2] > tl[idy[0]]:
        return None
    if zmin <= z:
        return None
    b = bo.copy()
    b[idx[0], idx[1] : -1] += yo[idy] - yo[idx]
    b[idy[0], idy[1] : -1] += yo[idy] - yo[idx]
    if np.any(b.T > k) or np.any(b < 0):
        return None
    return zmin - z