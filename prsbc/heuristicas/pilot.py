import numpy as np
import numpy.typing as npt
from prsbc.heuristicas.hcv import *
from prsbc.utilidades.constantes import *


def pilot_single(
    lm: np.intp,
    ro: npt.NDArray[np.intp],
    yo: npt.NDArray[np.int_],
    bo: npt.NDArray[np.int_],
    tt: npt.NDArray[np.int_],
    v: npt.NDArray[np.intp],
    ao: npt.NDArray[np.int_],
    p: npt.NDArray[np.int_],
    q: npt.NDArray[np.int_],
    t: npt.NDArray[np.int_],
    k: np.int_,
    tl: np.int_,
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    np.int_,
    np.float_,
]:
    r = ro.copy()
    y = yo.copy()
    b = bo.copy()
    a = ao.copy()
    u = r[lm]

    while lm < MAX_PARADAS:
        # Paso 3 (mejoraría eficiencia si sólo me centrase en los que ya escogí?)
        F = v[a != q]
        F = F[(tt[lm] + t[u, :][F] + TIEMPO_OPERACION + t[:, 0][F]) < tl]
        if F.size == 0:
            break

        # Paso 4
        bb = prehvc(u, tt[lm], tl, v, a, q, t)
        if b[lm] >= bb:
            F = F[a[F] < q[F]]
        if F.size == 0:
            break

        # Paso 5
        Vm = a[F] < q[F]
        Vp = a[F] > q[F]
        g = np.zeros(F.size, dtype=np.int_)
        g[Vp] = np.fmin(a[F][Vp] - q[F][Vp], min(k, bb) - b[lm])
        g[Vm] = np.fmin(q[F][Vm] - a[F][Vm], b[lm])

        # Paso 6

        gs = None
        zmin = np.inf
        vs = None
        for idv, nv in enumerate(v[F]):
            tt[lm + 1] = tt[lm] + t[u, nv] + TIEMPO_OPERACION
            r[lm + 1] = nv
            y[lm + 1] = g[idv] if (a[nv] > q[nv]) else -g[idv]
            av = a.copy()
            av[nv] -= y[lm + 1]
            b[lm + 1] = b[lm] + y[lm + 1]
            _, _, _, _, _, z = hcv_single(lm + 1, r, y, b, tt, v, av, p, q, t, k, tl)
            if z < zmin:
                vs = nv
                gs = idv
                zmin = z
        if g[gs] == 0:
            break

        lm = lm + 1
        tt[lm] = tt[lm - 1] + t[u, vs] + TIEMPO_OPERACION
        r[lm] = vs
        y[lm] = g[gs] if (a[vs] > q[vs]) else -g[gs]
        a[vs] = a[vs] - y[lm]
        b[lm] = b[lm - 1] + y[lm]
        u = vs

    # Paso 7
    tt[lm] = tt[lm] + t[u, 0]
    if b[-2] != 0:
        lmb = lm
        while True:
            # Paso 8
            rb = r[np.arange(1, lmb + 1)]
            # Esto es equivalente a al paso 8 del algoritmo original
            if len(rb) == 0:
                break
            vs = rb[p[rb] > q[rb]][-1]
            lmb = np.where(r == vs)[0][0]
            if y[lmb] > b[-2]:
                a[vs] += b[-2]

                y[lmb] -= b[-2]
                b[-2] = np.int_(0)
                break

            # Paso 9
            a[vs] += y[lmb]
            b[-2] -= y[lmb]
            y[lmb] = 0

            # no entiendo las condiciones de este paso para tt[l], dependen de lmb y lm[l]? solo de 1?
            if lm == 1:
                tt[lm] = np.int_(0)
            elif lmb == 1 and lmb < lm:
                tt[lm] += t[0, r[2]] - t[0, vs] - t[vs, r[2]]
            elif 1 < lmb and lmb < lm:
                tt[lm] += t[r[lmb - 1], r[lmb + 1]] - t[0, vs] - t[vs, r[lmb + 1]]
            else:
                tt[lm] += t[r[lmb - 1], 0] - t[r[lmb - 1], vs] - t[vs, 0]

            lmb = lmb - 1
            # print(f"quitando {vs}")
            y[lm:lmb] = y[lm + 1 : lmb + 1]
            r[lm:lmb] = r[lm + 1 : lmb + 1]  # debería estar bien, pero no se
            if b[-2] <= 0:
                break
            lm = lm - 1

    z = (
        np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
        + W_2 * np.sum(np.abs(y))
        + W_3 * tt[-2]
    )

    return (r, y, b, a, tt, z)


def pilot_fleet(
    v: npt.NDArray[np.intp],
    p: npt.NDArray[np.int_],
    q: npt.NDArray[np.int_],
    t: npt.NDArray[np.int_],
    k: npt.NDArray[np.int_],
    tl: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int_], np.float_]:
    """Calcula la ruta óptima a realizar por cada vehículo.

    Args:
        p (npt.ArrayLike): Punto de partida.
        q (npt.ArrayLike): Punto de llegada.
        C (npt.ArrayLike): Costo de viaje.
        t (npt.ArrayLike): Tiempo de viaje.

    Returns:
        npt.ArrayLike: Ruta a realizar y bicicletas a depositar/recoger en cada parada.
    """
    # Paso 1
    a = p.copy()
    # la gestión de l es problema del bucle for
    r = np.zeros((NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.intp)
    y = np.zeros((NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.int_)
    b = np.zeros((NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.int_)
    tt = np.zeros((NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.int_)

    # Paso 2
    for l in np.arange(1, NUM_VEHICULOS + 1, dtype=np.intp):
        r[l, :], y[l, :], b[l, :], a, tt[l, :], _ = pilot_single(
            np.intp(0),
            np.zeros(MAX_PARADAS + 2, dtype=np.intp),
            np.zeros(MAX_PARADAS + 2, dtype=np.int_),
            np.zeros(MAX_PARADAS + 2, dtype=np.int_),
            np.zeros(MAX_PARADAS + 2, dtype=np.int_),
            v,
            a,
            p,
            q,
            t,
            k[l],
            tl[l],
        )

    z = (
        np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
        + W_2 * np.sum(np.sum(np.abs(y)))
        + W_3 * np.sum(tt[:, -2])
    )

    return (r, y, b, a, tt, z)