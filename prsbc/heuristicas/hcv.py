import numpy as np
import numpy.typing as npt
from prsbc.utilidades.constantes import *
from prsbc.utilidades.cronometrado import cronometrando


def prehvc(
    r: np.intp,
    tp: np.int_,
    tl: np.int_,
    v: npt.NDArray[np.intp],
    a: npt.NDArray[np.int_],
    q: npt.NDArray[np.int_],
    t: npt.NDArray[np.int_],
) -> np.int_:
    """Calcula cuantas bicicletas se pueden dejar antes de alcanzar el tiempo
    límite y asumiendo que hubiese infinitas en el vehículo.

    Args:
        r (np.intp): Estación de partida.
        tp (np.int_): Tiempo de partida.
        tl (np.int_): Tiempo límite.
        a (npt.NDArray[np.intp]):
            Cantidad actual de bicicletas por estación.
        q (npt.NDArray[np.intp]):
            Cantidad deseada de bicicletas por estación.
        t (npt.NDArray[np.intp]):
            Tiempos de translado del vehículo entre estaciones.

    Returns:
        int_: Número de bicicletas depositables.

    """
    # Paso 1
    u = np.intp(r)
    vs: np.intp
    Fm = v[a < q]
    Fm = Fm[(tp + t[u, :][Fm] + TIEMPO_OPERACION + t[:, 0][Fm]) < tl]
    g = q - a
    bb = np.int_(0)
    # Paso 2
    while True:
        if Fm.size == 0:
            break
        else:
            vs = Fm[np.argmax(g[Fm] / (t[u, Fm] + TIEMPO_OPERACION))]
            Fm = Fm[Fm != vs]
            tp = tp + t[u, vs] + TIEMPO_OPERACION
            u = vs
            bb += g[vs]
            Fm = Fm[(tp + t[u, :][Fm] + TIEMPO_OPERACION + t[:, 0][Fm]) < tl]
    return bb


def hcv_single(
    lm: np.intp,
    ro: npt.NDArray[np.intp],
    yo: npt.NDArray[np.int_],
    bo: npt.NDArray[np.int_],
    tto: npt.NDArray[np.int_],
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
    npt.NDArray[np.int_],
    np.float_,
]:
    """Heurística de construcción y mejora para el problema de recogida y
    entrega de bicicletas capáz de empezar en una estación cualquiera con
    parte del algoritmo ya ejecutado.

    Args:
        lm (np.intp): Índice de la primera parada a retomar.
        ro (npt.NDArray[np.intp]):
            Ruta a retomar.
        yo (npt.NDArray[np.int_]):
            Cantidad de bicicletas recogidas o depositadas a retomar.
        bo (npt.NDArray[np.int_]):
            Cantidad de bicicletas en el vehículo a retomar.
        tto (npt.NDArray[np.int_]):
            Tiempo de llegada a la estación de recogida a retomar.
        v (npt.NDArray[np.intp]):
            Estaciones visitables.
        ao (npt.NDArray[np.int_]):
            Cantidad de bicicletas en cada estación a retomar.
        p (npt.NDArray[np.int_]):
            Cantidad de bicicletas en cada estación previas a la ruta.
        q (npt.NDArray[np.int_]):
            Cantidad deseada de bicicletas por estación.
        t (npt.NDArray[np.int_]):
            Tiempos de translado del vehículo entre estaciones.
        k (np.int_):
            Capacidad del vehículo.
        tl (np.int_):
            Tiempo límite para el vehículo.

    Returns:
        z (npt.NDArray[np.intp]):
            Ruta del vehículo.
        y (npt.NDArray[np.int_]):
            Cantidad de bicicletas recogidas o depositadas en cada parada.
        b (npt.NDArray[np.int_]):
            Cantidad de bicicletas en el vehículo en cada parada.
        a (npt.NDArray[np.int_]):
            Cantidad de bicicletas en cada estación al finalizar la ruta.
        tt (npt.NDArray[np.int_]):
            Tiempo de llegada a cada estación.
        z (np.float_),
            Valor de la función objetivo.
    """
    r = ro.copy()
    y = yo.copy()
    b = bo.copy()
    a = ao.copy()
    tt = tto.copy()
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
        g[Vp] = np.fmin(a[F][Vp] - q[F][Vp], min(k, bb) - b[lm]) # type: ignore
        g[Vm] = np.fmin(q[F][Vm] - a[F][Vm], b[lm])

        # Paso 6
        gs = np.argmax(g / (t[u, :][F] + TIEMPO_OPERACION))

        vs = v[F][gs]
        if g[gs] == 0:
            break

        lm = lm + 1
        tt[lm] = tt[lm - 1] + t[u, vs] + TIEMPO_OPERACION
        r[lm] = vs
        y[lm] = g[gs] if (a[vs] > q[vs]) else -g[gs]
        a[vs] -= y[lm]
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
            if len(rb[p[rb] > q[rb]]) == 0:
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

            if lm == 1:
                tt[lm] = np.int_(0)
            elif lmb == 1 and lmb < lm:
                tt[lm] += t[0, r[2]] - t[0, vs] - t[vs, r[2]]
            elif 1 < lmb and lmb < lm:
                tt[lm] += t[r[lmb - 1], r[lmb + 1]] - t[0, vs] - t[vs, r[lmb + 1]]
            else:
                tt[lm] += t[r[lmb - 1], 0] - t[r[lmb - 1], vs] - t[vs, 0]

            lmb = lmb - 1
            y[lm:lmb] = y[lm + 1 : lmb + 1]
            r[lm:lmb] = r[lm + 1 : lmb + 1]
            if b[-2] <= 0:
                break
            lm = lm - 1

    z = (
        np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
        + W_2 * np.sum(np.abs(y))
        + W_3 * tt[-2]
    )

    return r, y, b, a, tt, z


def hcv_fleet(
    v: npt.NDArray[np.intp],
    p: npt.NDArray[np.int_],
    q: npt.NDArray[np.int_],
    t: npt.NDArray[np.int_],
    k: npt.NDArray[np.int_],
    tl: npt.NDArray[np.int_],
    idx: tuple[np.intp, np.intp] = (np.intp(0), np.intp(0)),
    ro: npt.NDArray[np.intp] = np.zeros(
        (NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.intp
    ),
    yo: npt.NDArray[np.int_] = np.zeros(
        (NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.int_
    ),
    ao: npt.NDArray[np.int_] | None = None,
    bo: npt.NDArray[np.int_] = np.zeros(
        (NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.int_
    ),
    tto: npt.NDArray[np.int_] = np.zeros(
        (NUM_VEHICULOS + 1, MAX_PARADAS + 2), dtype=np.intp
    ),
) -> tuple[
    npt.NDArray[np.intp],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    np.float_,
]:
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
    l, lm = idx
    if ao is None:
        a = p.copy()
    else:
        a = p.copy()
        a[ro[1:l, :]] = ao[ro[1:l, :]]
        a[ro[1 : l + 1, 1 : lm + 1]] = ao[ro[1 : l + 1, 1 : lm + 1]]
    # la gestión de l es problema del bucle for
    r = ro.copy()
    y = yo.copy()
    b = bo.copy()
    tt = tto.copy()

    r[l, :], y[l, :], b[l, :], a, tt[l, :], z = hcv_single(
        lm,
        r[l, :],
        y[l, :],
        b[l, :],
        tt[l, :],
        v,
        a,
        p,
        q,
        t,
        k[l],
        tl[l],
    )

    for l in np.arange(l + 1, NUM_VEHICULOS + 1, dtype=np.intp):
        r[l, :], y[l, :], b[l, :], a, tt[l, :], _ = hcv_single(
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
            k[l],  # type: ignore
            tl[l], # type: ignore
        )

    z = (
        np.sum(np.abs(np.subtract(a, q, dtype=np.int_)))
        + W_2 * np.sum(np.sum(np.abs(y)))
        + W_3 * np.sum(tt[:, -2])
    )

    return (r, y, b, a, tt, z)
