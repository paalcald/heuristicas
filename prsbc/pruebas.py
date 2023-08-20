import math
import json
import numpy as np
from prsbc.heuristicas.hcv import *
from prsbc.utilidades.constantes import *

def basica():
    file_estaciones202206 = open("./prsbc/datos/202206.json", "r")
    estaciones_k = json.loads(file_estaciones202206.readline())

    # de estos extraemos los valores relevantes para las estaciones
    # el 0-esimo elemento es la base de operaciones
    p = np.array(
        [0] + [estacion["dock_bikes"] for estacion in estaciones_k["stations"]],
        dtype=np.int_,
    )
    C = np.array(
        [0] + [estacion["total_bases"] for estacion in estaciones_k["stations"]],
        dtype=np.int_,
    )
    # Asignamos nuevas id's por que las estaciones de bicimad no tienen id's consecutivas
    v = np.arange(0, p.size, dtype=np.intp)
    # Asumimos que el valor deseado de bicicletas es el 66% de la capacidad de la estación
    q = (C * 0.50).astype(np.int_)
    # Asignamos tiempos aleatorios para llegar de una estación a otra,
    # podriamos usar la api de google maps para obtener tiempos reales
    x = np.zeros((len(p), 2))
    x[0, :] = [-3.675557, 40.391240]
    x[1:, 0] = np.array(
        [estacion["longitude"] for estacion in estaciones_k["stations"]]
    )
    x[1:, 1] = np.array([estacion["latitude"] for estacion in estaciones_k["stations"]])
    # lt.plot(x[1:, 0], x[1:, 1], "o")
    # plt.show()
    alpha = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    dist = alpha * 2 * RADIO_TERRESTRE * math.pi / 180

    t = (dist / 8).astype(np.int_)
    N = np.argsort(t, axis=1)

    k = np.full(NUM_VEHICULOS + 1,
                CAPACIDAD_VEHICULO,
                dtype=np.int_)
    k[0] = 0
    tl = np.full(NUM_VEHICULOS + 1,
                 JORNADA_EN_HORAS * 60 * 60,
                 dtype=np.int_)
    #
    r, y, b, a, tt, z = hvc_fleet(v, p, q, t, k, tl)
    print(f"r ={r[1:,:]} \ny ={y[1:,:]}\nz = {z}\ntt={tt[1:, :]}")
    #r, y, b, a, tt, z = pilotFlota(v, p, q, t, k, tl)
    #print(f"r ={r[1:,:]} \ny ={y[1:,:]}\nz = {z}\ntt={tt[1:,:]}")
    # idx = (1, 3)
    # idy = (3, 1)
    # r, y, b, tt, z = swap(idx, idy, r, y, b, tt, a, q, t)
    #r, y, b, tt, z = VND(r, y, b, tt, a, q, t, k, tl, 50, N)
    #print(f"r ={r[1:,:]} \ny ={y[1:,:]}\nz = {z}\ntt={tt[1:,:]}")