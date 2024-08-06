import numpy as np
import pandas as pd

import SatelliteData


# -*- coding: utf-8 -*-
# @Time    : 2024/7/17 20:23
# @Author  : Zhang Huan
# @Email   : johnhuan@whu.edu.cn
# QQ       : 248404941
# @File    : d.py

def readSatellite(filepath):
    APPROX_POSITION = []  # 测站概略坐标
    satNumsListInEachEpoch = []  # 每个历元下的卫星数
    epochs = []  # 所有的历元
    satelliteAllData = list()  # 所有的卫星数据
    # 历元数组
    with open(filepath, 'r') as f:
        while True:
            currentLine = f.readline()  # 读一行数据
            if currentLine.startswith('APPROX_POSITION'):
                APPROX_POSITION.append(float(currentLine[16:31]))
                APPROX_POSITION.append(float(currentLine[32:47]))
                APPROX_POSITION.append(float(currentLine[48:63]))
            elif currentLine.startswith('Satellite'):  # 当卫星开始时，向下读取指定行从而获取指定颗卫星的观测数据
                clArr = currentLine.split(',')
                satNumsInEachEpoch = int(clArr[0][17:])
                epoch = int(clArr[1][12:])
                satNumsListInEachEpoch.append(satNumsInEachEpoch)
                epochs.append(epoch)
                satelliteDatasAtEachEpoch = list()
                for i in range(0, satNumsInEachEpoch):
                    cl = f.readline()  # 读一行数据
                    clArr = cl.split(',')
                    sd = SatelliteData.SatelliteData()
                    sd.PRN = str(clArr[0])
                    sd.SatpositionX = float(clArr[1])
                    sd.SatpositionY = float(clArr[2])
                    sd.SatpositionZ = float(clArr[3])
                    sd.SatClock = float(clArr[4])
                    sd.Elevation = float(clArr[5])
                    sd.CL = float(clArr[6])
                    sd.TropDelay = float(clArr[7])
                    satelliteDatasAtEachEpoch.append(sd)
                satelliteAllData.append(satelliteDatasAtEachEpoch)
            if currentLine == '':  # 到达末尾了，退出死循环
                break
    return satelliteAllData, APPROX_POSITION, epochs


if __name__ == '__main__':
    Rho_P_0_2 = 0.04
    filepath = './data/GPS卫星数据.txt'
    satelliteAllData, APPROX_POSITION, epochs = readSatellite(filepath)
    X, Y, Z, mxs, mys, mzs, PDOPS = [], [], [], [], [], [], []
    satNums = []
    for eachEphSats in satelliteAllData:
        l, m, n, P, R_0, dt_sat, dtrop, P_i, L = [], [], [], [], [], [], [], [], []
        satNum = len(eachEphSats)
        satNums.append(satNum)
        for sat in eachEphSats:
            rho_x = sat.SatpositionX - APPROX_POSITION[0]
            rho_y = sat.SatpositionY - APPROX_POSITION[1]
            rho_z = sat.SatpositionZ - APPROX_POSITION[2]
            R_0_i = np.sqrt(rho_x ** 2 + rho_y ** 2 + rho_z ** 2)
            l_i = (sat.SatpositionX - APPROX_POSITION[0]) / R_0_i
            m_i = (sat.SatpositionY - APPROX_POSITION[1]) / R_0_i
            n_i = (sat.SatpositionZ - APPROX_POSITION[2]) / R_0_i
            l.append(l_i)
            m.append(m_i)
            n.append(n_i)
            P.append(sat.CL)
            R_0.append(R_0_i)
            dt_sat.append(sat.SatClock)
            dtrop.append(sat.TropDelay)
            L.append(sat.CL - R_0_i + sat.SatClock - sat.TropDelay)
            P_i.append(np.sin(np.deg2rad(sat.Elevation)) / Rho_P_0_2)
        B = np.zeros((satNum, 4))
        B[:, 0] = np.array(l)
        B[:, 1] = np.array(m)
        B[:, 2] = np.array(n)
        B[:, 3] = np.array([-1])
        P = np.zeros((satNum, satNum))
        for i in range(0, satNum):
            P[i, i] = P_i[i]
        L = np.array(L)
        BT = np.transpose(B)
        BTP = np.dot(BT, P)
        BTPB = np.dot(BTP, B)
        # BTPL = np.dot(BTP, L)
        # BTPBN = 0
        BTPBN = np.linalg.inv(BTPB)
        BTPBNBT = np.dot(BTPBN, BT)
        BTPBNBTP = np.dot(BTPBNBT, P)
        BTPBNBTPL = np.dot(BTPBNBTP, L)
        dxs = -BTPBNBTPL
        V = np.dot(B, dxs) + L
        VTP = np.dot(np.transpose(V), P)
        VTPV = np.dot(VTP, V)
        m0 = np.sqrt(VTPV / (satNum - 4))
        mx = m0 * BTPBN[0][0]
        my = m0 * BTPBN[1][1]
        mz = m0 * BTPBN[2][2]
        mxs.append(mx)
        mys.append(my)
        mzs.append(mz)
        PDOP = np.sqrt(BTPBN[0][0] * BTPBN[0][0] + BTPBN[1][1] * BTPBN[1][1] + BTPBN[2][2] * BTPBN[2][2])
        PDOPS.append(PDOP)
        X.append(dxs[0] + APPROX_POSITION[0])
        Y.append(dxs[1] + APPROX_POSITION[1])
        Z.append(dxs[2] + APPROX_POSITION[2])
        dxs = 0

        # 整理输出数据
    df = pd.DataFrame({
        'epoch': epochs,
        'satNum': satNums,
        'X/(m)': X,
        'Y/(m)': Y,
        'Z/(m)': Z,
        'σx/m': mxs,
        'σy/m': mys,
        'σz/m': mzs,
        'PDOP': PDOPS
    })
    df.to_csv('./data/result.csv')
