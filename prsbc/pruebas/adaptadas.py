import json
from prsbc.utilidades.constantes import *
from prsbc.heuristicas.hcv import hcv_fleet as HCV
from prsbc.heuristicas.pilot import pilot_fleet as PILOT
from prsbc.heuristicas.vnd import vnd_extended as VND
from prsbc.heuristicas.vnd import vnd_basic as VND_BASIC
from prsbc.utilidades.calculo_tiempos import create_t
import numpy as np
from time import time

def main():
    # Parametros
    NUM_ESTACIONES = 100
    NUM_PRUEBAS = 10
    k = np.full(NUM_VEHICULOS + 1, CAPACIDAD_VEHICULO, dtype=np.int_)
    k[0] = np.int_(0)
    tl = np.full(NUM_VEHICULOS + 1, JORNADA_EN_HORAS * 60 * 60, dtype=np.int_)
    tl[0] = np.int_(0)

    # Para guardar los resultados
    resultados_hcv = []
    resultados_pilot = []
    resultados_vnd = []
    resultados_vndpilot = []
    resultados_vndbasic = []
    resultados_vndbasicpilot = []
    t_hcv = 0
    t_pilot = 0
    t_vnd = 0
    t_vndpilot = 0
    t_vndbasic = 0
    t_vndbasicpilot = 0

    i = 0
    file_estaciones202206 = open("./prsbc/datos/202206.json")
    estaciones_k = json.loads(file_estaciones202206.readline())
    po = np.array([0] + [estacion['dock_bikes']
                        for estacion in estaciones_k['stations']], dtype=np.int_)
    C = np.array([0] + [estacion['total_bases']
                        for estacion in estaciones_k['stations']], dtype=np.int_)
    vo = np.arange(0, po.size)
    qo = (C * 2/3).astype(np.int_)
    while i < NUM_PRUEBAS:
        
        try:
        
            # Datos
            t, random_idx = create_t(NUM_ESTACIONES)
            N = np.argsort(t, axis=1)
            
            # Estaciones aleatorias
            p = po[random_idx]
            v = np.arange(0, NUM_ESTACIONES, dtype=np.intp)
            q = qo[random_idx]
            
            # Algoritmos
            tt = time()
            r, y, b, a, ttt, z = HCV(v,p,q,t,k,tl)
            t_hcv += time() - tt
            resultados_hcv.append(z)

            tt = time()
            _, _, _, _, z = VND_BASIC(r, y, b, ttt, a, q, t, k, tl, 10, N)
            t_vndbasic += time() - tt
            resultados_vndbasic.append(z)

            tt = time()
            _, _, _, _, _, z = VND(r, y, b, ttt, z, v, p, a, q, t, k, tl, 10, N)
            t_vnd += time() - tt
            resultados_vnd.append(z)

            tt = time()
            r, y, b, a, ttt, z = PILOT(v, p, q, t, k, tl)
            t_pilot += time() - tt
            resultados_pilot.append(z)

            tt = time()
            _, _, _, _, z = VND_BASIC(r, y, b, ttt, a, q, t, k, tl, 10, N)
            t_vndbasicpilot += time() - tt
            resultados_vndbasicpilot.append(z)

            tt = time()
            _, _, _, _, _, z = VND(r, y, b, ttt, z, v, p, a, q, t, k, tl, 10, N)
            t_vndpilot += time() - tt
            resultados_vndpilot.append(z)


            i += 1
            print(i) # para saber por donde vamos
        
        except:
            
            print("Error")

    # Resultados
    print("\n\nHCV\n")
    print("Media:", np.mean(resultados_hcv), "\nDesviación típica:",
        np.std(resultados_hcv), "\nMínimo:", min(resultados_hcv), "\nMáximo:",
        max(resultados_hcv), "\nTiempo medio ejecución:", t_hcv/NUM_PRUEBAS)
    print("\nVND BASICO desde HCV\n")
    print("Media:", np.mean(resultados_vndbasic), "\nDesviación típica:",
        np.std(resultados_vndbasic), "\nMínimo:", min(resultados_vndbasic),
        "\nMáximo:", max(resultados_vndbasic), "\nTiempo medio ejecución:",
        t_vndbasic/NUM_PRUEBAS)
    print("\nVND(HCV) desde HCV\n")
    print("Media:", np.mean(resultados_vnd), "\nDesviación típica:",
        np.std(resultados_vnd), "\nMínimo:", min(resultados_vnd),
        "\nMáximo:", max(resultados_vnd), "\nTiempo medio ejecución:",
        t_vnd/NUM_PRUEBAS)
    print("\nPILOT\n")
    print("Media:", np.mean(resultados_pilot), "\nDesviación típica:",
        np.std(resultados_pilot), "\nMínimo:", min(resultados_pilot),
        "\nMáximo:", max(resultados_pilot), "\nTiempo medio ejecución:",
        t_pilot/NUM_PRUEBAS)
    print("\nVND BASICO desde PILOT\n")
    print("Media:", np.mean(resultados_vndbasicpilot), "\nDesviación típica:",
        np.std(resultados_vndbasicpilot), "\nMínimo:", min(resultados_vndbasicpilot),
        "\nMáximo:", max(resultados_vndbasicpilot), "\nTiempo medio ejecución:",
        t_vndbasicpilot/NUM_PRUEBAS)
    print("\nVND(HCV) desde PILOT\n")
    print("Media:", np.mean(resultados_vndpilot), "\nDesviación típica:",
        np.std(resultados_vndpilot), "\nMínimo:", min(resultados_vndpilot),
        "\nMáximo:", max(resultados_vndpilot), "\nTiempo medio ejecución:",
        t_vndpilot/NUM_PRUEBAS)
