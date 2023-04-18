#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate


class Function:
    """This class loads a function from a spread of sorted data.
    It has the 'eval' method, which evaluates the function on any number as long as it is in the function domain. For now it only uses a linear interpolation to calculate the result.
    It must be initialized with a tuple of X and Y data lists. e.g.:

    testFunction = Function( (pointsListX, pointsListY))"""

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
        """Linear interpolation method, the interpolators must be written in such a way that they only need one X value to produce one Y value."""
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
        """Plots the contained function"""
        fig, ax = plt.subplots()
        ax.plot(self.xDataPoints, self.yDataPoints)
        ax.set(xlabel=xLabel, ylabel=yLabel)
        ax.grid()
        plt.show()

def clean_gromacs_garbage(path=os.getcwd()):
    """Deletes backups left by Gromacs"""
    garbagePattern = re.compile(r"#([\w\d.]+)#")
    for file in os.listdir(path):
        if garbagePattern.match(file):
            os.remove(os.path.join(path, file))
            print(os.path.join(path, file), "removed")

def get_overlap(function1, function2):
    """Receives two Function objects, and returns the lower and upper bounds where the two domains overlap. Useful for generating a third function from two."""
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
    """From two free energy functions at different temperatures produce an enthalpy function for the same process."""
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

def show_umbrella_plot(profileFilename, histogramFilename):
    """Displays the profile graph and histograms on the same graph. Useful to determine if the calculation is missing windows."""
    figure = plt.figure()

    histogramsData = parseXVG(histogramFilename)
    histoPlot = figure.add_subplot(111)
    for histogramNum in range(1, len(histogramsData)):
        histoPlot.fill_between(
            histogramsData[0], 0, histogramsData[histogramNum], color="grey", alpha=0.35
        )
        histoPlot.set_xlabel("Distance from bilayer center [nm]")
        histoPlot.set_ylabel("Population")

    profileData = parseXVG(profileFilename)
    profilePlot = figure.add_subplot(111, sharex=histoPlot, frameon=False)
    profilePlot.plot(profileData[0], profileData[1])
    profilePlot.yaxis.tick_right()
    profilePlot.yaxis.set_label_position("right")
    profilePlot.set_ylabel("Mean force potential [kj/mol]")
    profilePlot.grid()
    plt.show()

def generate_tpr_list_file(path, tprListFile="tpr_files.dat"):
    """Generates a tpr_files.dat file which contains every tpr file in the directory. Useful for umbrella sampling with GROMACS."""
    windowsList = []
    pattern = re.compile(r"umbrella([\w.]+).gro")
    for file in os.listdir(path):
        if pattern.match(file):
            windowsList.append(pattern.findall(file)[0])
    try:
        os.remove(path + tprListFile)
    except:
        print("No previous tpr file found")
    outputFile = open(path + tprListFile, "w+")
    for window in windowsList:
        print("umbrella" + window + ".tpr", file=outputFile)
    outputFile.close()


def generate_pullf_list_file(path, pullfListFile="pullf_files.dat"):
    """Generates a pullf_files.dat file which contains every pullf file in the directory. Useful for umbrella sampling with GROMACS."""
    windowsList = []
    pattern = re.compile(r"umbrella([\w.]+).gro")
    for file in os.listdir(path):
        if pattern.match(file):
            windowsList.append(pattern.findall(file)[0])
    try:
        os.remove(path + pullfListFile)
    except:
        print("No provious pullf list found")
    outputFile = open(path + pullfListFile, "w+")
    for window in windowsList:
        print("pullf_umbrella" + window + ".xvg", file=outputFile)
    outputFile.close()

def list_finished_runs(path=os.getcwd()):
    windowsList = []
    pattern = re.compile(r"umbrella([\w.]+).gro")
    for file in os.listdir(path):
        if pattern.match(file):
            windowsList.append(pattern.match(file)[1])
    return windowsList

def xvg_to_dataframe(xvgFilename):
    """Returns a dataframe from a XVG file. The filename of the XVG file needs to be provided"""
    # Transformar el archivo xvg en un dataFrame
    xvgArray = np.loadtxt(xvgFilename, comments=["#", "@"])
    xvgDataFrame = pd.DataFrame(xvgArray)
    xvgDataFrame = xvgDataFrame.set_index(0)
    # Buscar el nombre de las columnas en el metadato del archivo xvg
    columnNames = []
    if len(xvgDataFrame.columns) == 1:
        columnNamePattern = re.compile(r"@[\s]+title\s\"([\w]+)")
    else:
        columnNamePattern = re.compile(r"@\ss\d\slegend\s\"([\w\s]+)")
    xvgFileData = open(xvgFilename, "r")
    while len(columnNames) < (len(xvgDataFrame.columns)):
        line = xvgFileData.readline()
        if line.startswith("#"):
            continue
        elif line.startswith("@"):
            if columnNamePattern.match(line):
                columnNames.append(columnNamePattern.findall(line)[0])
        else:
            xvgFileData.close()
            columnNames = [str(i + 1) for i in range(len(xvgDataFrame.columns))]
            break
    xvgFileData.close()
    xvgDataFrame.columns = columnNames
    return xvgDataFrame
