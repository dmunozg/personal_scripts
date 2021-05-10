#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import pandas as pd
from scipy import interpolate


class Function:
    """Esta clase carga una función a partir de una disperción de datos ordenados.
    Posee el método 'eval', que evalúa la función en cualquier número mientras
    esté en el dominio de la función. Por ahora solo usa una interpolación lineal
    para calcular el resultado.
    Se debe inicializar con una tupla de listas de datos X e Y. e.g.:

    funcion_de_prueba = Function( (listaDePuntosX, listaDePuntosY))"""

    def __init__(self, plotData, interpolator="linear"):
        self.interpolator = interpolator
        self.xDataPoints = []
        self.yDataPoints = []
        self.xDataPoints, self.yDataPoints = plotData[0], plotData[1]
        # Generar representación de splines cúbicos
        if interpolator == "spline":
            self.splineRepresentation = interpolate.splrep(self.xDataPoints, self.yDataPoints)

    def eval(self, xValue):
        # Revisar si esta fuera del dominio
        if not min(self.xDataPoints) < xValue < max(self.xDataPoints):
            print("ERROR: trying to evaluate outside domain")
            return False
        # Revisar si coincide con alguno de los puntos
        index = 0
        while index < len(self.xDataPoints):
            if xValue == self.xDataPoints[index]:
                return self.yDataPoints[index]
            index += 1
        # Si no coincide ningún valor, interpolar.
        if self.interpolator == "linear":
            return self.linear_interpol(xValue)
        elif self.interpolator == "spline":
            return self.spline_interpol(xValue)
        else:
            print("ERROR: Unknown interpolator")
            return False

    def linear_interpol(self, xValue):
        """Método de interpolación lineal, los interpoladores se deben escribir
        de manera que sólo nececiten un valor X para producir un valor Y"""
        # Encontrar los valores x0 y x1 mas cercanos a xValue y sus respectivas
        # imágenes
        index = 1
        while index < len(self.xDataPoints):
            if self.xDataPoints[index] > xValue:
                x0 = self.xDataPoints[index - 1]
                y0 = self.yDataPoints[index - 1]
                x1 = self.xDataPoints[index]
                y1 = self.yDataPoints[index]
                break
            else:
                index += 1
                continue
        return y0 + (xValue - x0) * (y1 - y0) / (x1 - x0)

    def spline_interpol(self, xValue):
        return interpolate.splev(xValue, self.splineRepresentation)

    def show(self, xLabel=None, yLabel=None):
        """Grafica la función contenida"""
        fig, ax = plt.subplots()
        ax.plot(self.xDataPoints, self.yDataPoints)
        ax.set(xlabel=xLabel, ylabel=yLabel)
        ax.grid()
        plt.show()

def parseXVG(xvgFile):
    """Lee un archivo XVG, y devuelve una lista con cada columna de datos presente en
    el archivo"""
    sourceFile = open(xvgFile, "r")
    columns = []
    for line in sourceFile:
        if line.startswith("#") or line.startswith("@"):
            continue
        lineContentStr = line.split()
        lineContentFloat = list(map(float, lineContentStr))
        columns.append(lineContentFloat)
    sourceFile.close()
    return list(map(list, zip(*columns)))  # Esto transpone la lista de listas

def xvg_to_dataframe(xvgFile):
    """Returns a dataframe from a XVG file. The filename of the XVG file needs to be provided"""
    dataColumns = parseXVG(xvgFile)
    columnNames = []
    if len(dataColumns) == 2:
        columnNamePattern = re.compile(r"@[\s]+title\s\"([\w]+)")
    else:
        columnNamePattern = re.compile(r"@\ss\d\slegend\s\"([\w\s]+)")
    xvgFileData = open(xvgFile, "r")
    while len(columnNames) < (len(dataColumns) - 1):
        line = xvgFileData.readline()
        if line.startswith("#"):
            continue
        elif line.startswith("@"):
            if columnNamePattern.match(line):
                columnNames.append(columnNamePattern.findall(line)[0])
        else:
            xvgFileData.close()
            logging.warning("Could not find enough matches in file {}".format(xvgFile))
            logging.debug(
                "{} of {} matches were found.".format(
                    len(columnNames), len(dataColumns) - 1
                )
            )
            columnNames = [str(i + 1) for i in range(len(dataColumns) - 1)]
            break
    xvgFileData.close()
    return pd.DataFrame(
        list(zip(*dataColumns[1:])), columns=columnNames, index=dataColumns[0]
    )

def clean_gromacs_garbage(path=os.getcwd()):
    """Deletes backups left by Gromacs"""
    garbagePattern = re.compile(r"#([\w\d.]+)#")
    for file in os.listdir(path):
        if garbagePattern.match(file):
            os.remove(os.path.join(path, file))
            print(os.path.join(path, file), "removed")

def get_overlap(function1, function2):
    """Recibe dos objetos Function, y devuelve los límites inferior y superior en que
    ambos dominios se superponen. Útil para generar una tercera función a partir de dos."""
    if min(function1.xDataPoints) < min(function2.xDataPoints):
        xMin = min(function2.xDataPoints)
    else:
        xMin = min(function1.xDataPoints)
    if max(function1.xDataPoints) < max(function2.xDataPoints):
        xMax = max(function1.xDataPoints)
    else:
        xMax = max(function2.xDataPoints)
    return [xMin, xMax]


def calculate_enthalpy_plot(lowTempFunc, highTempFunc, deltaTemp, nPoints=200):
    """A partir de dos funciones de Energía libre a diferentes temperaturas produce una
    función de entalpía para el mismo proceso"""
    xLowLimit, xHighLimit = get_overlap(lowTempFunc, highTempFunc)
    deltaX = (xHighLimit - xLowLimit) / nPoints
    xValues = []
    enthalpyValues = []
    currentX = xLowLimit
    while currentX <= xHighLimit:
        currentX += deltaX
        xValues.append(currentX)
        enthalpyValues.append(
            -(highTempFunc.eval(currentX) - lowTempFunc.eval(currentX)) / deltaTemp
        )
    return Function([xValues, enthalpyValues])

