import json
from prsbc.heuristicas.hcv import hcv_fleet as HCV
from prsbc.heuristicas.pilot import pilot_fleet as PILOT
from prsbc.heuristicas.vnd import vnd_extended as VND
from prsbc.utilidades.calculo_tiempos import create_t
import numpy as np
from time import time

def main():
    # Parametros
    NUM_ESTACIONES = 50
    NUM_PRUEBAS = 10
    k = np.full(5, 20, dtype=np.int_)
    k[0] = np.int_(0)
    tl = np.full(5, 3 * 60 * 60, dtype=np.int_)
    tl[0] = np.int_(0)

    # Para guardar los resultados
    resultados_hcv = []
    resultados_pilot = []
    resultados_vnd = []
    t_hcv = 0
    t_pilot = 0
    t_vnd = 0

    i = 0

    while i < NUM_PRUEBAS:
        
        try:
        
            # Datos
            file_estaciones202206 = open("./prsbc/datos/202206.json")
            estaciones_k = json.loads(file_estaciones202206.readline())
            p = np.array([0] + [estacion['dock_bikes']
                                for estacion in estaciones_k['stations']], dtype=np.int_)
            C = np.array([0] + [estacion['total_bases']
                                for estacion in estaciones_k['stations']], dtype=np.int_)
            v = np.arange(0, p.size)
            q = (C * 2/3).astype(np.int_)
            t, random_idx = create_t(NUM_ESTACIONES)
            N = np.argsort(t, axis=1)
            
            # Estaciones aleatorias
            p = p[random_idx]
            v = np.arange(0, NUM_ESTACIONES, dtype=np.intp)
            q = q[random_idx]
            
            # Algoritmos
            tt = time()
            r, y, b, a, ttt, z = HCV(v,p,q,t,k,tl)
            t_hcv += time() - tt
            resultados_hcv.append(z)

            tt = time()
            r, y, b, a, _, z = VND(r, y, b, ttt, z, v, p, a, q, t, k, tl, 10, N)
            t_vnd += time() - tt
            resultados_vnd.append(z)

            tt = time()
            r, y, b, a, _, z = PILOT(v, p, q, t, k, tl)
            t_pilot += time() - tt
            resultados_pilot.append(z)

            i += 1
            print(i) # para saber por donde vamos
        
        except:
            
            print("Error")

    # Resultados
    print("\n\nHCV\n")
    print("Media:", np.mean(resultados_hcv), "\nDesviación típica:",
        np.std(resultados_hcv), "\nMínimo:", min(resultados_hcv), "\nMáximo:",
        max(resultados_hcv), "\nTiempo medio ejecución:", t_hcv/NUM_PRUEBAS)
    print("\nVND\n")
    print("Media:", np.mean(resultados_vnd), "\nDesviación típica:",
        np.std(resultados_vnd), "\nMínimo:", min(resultados_vnd),
        "\nMáximo:", max(resultados_vnd), "\nTiempo medio ejecución:",
        t_vnd/NUM_PRUEBAS)
    print("\nPILOT\n")
    print("Media:", np.mean(resultados_pilot), "\nDesviación típica:",
        np.std(resultados_pilot), "\nMínimo:", min(resultados_pilot),
        "\nMáximo:", max(resultados_pilot), "\nTiempo medio ejecución:",
        t_pilot/NUM_PRUEBAS)
