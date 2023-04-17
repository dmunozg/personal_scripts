#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scripts to transform and analyze NMR data obtained from a Bruker NMR spectrometer
"""

# standard library imports
import os
# required third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nmrglue as ng

def bruker_to_pd(fidDirectory, autoPhase=False) -> pd.DataFrame:
    # read the fid file
    dic, data = ng.bruker.read(fidDirectory)
    procData = ng.bruker.remove_digital_filter(dic, data)
    # zero fill the fid function
    dataSize = dic["procs"]["SI"]
    procData = ng.proc_base.zf_size(procData, dataSize)
    # Apodize the fid function
    # TODO: Line broadening should be a parameter
    procData = ng.proc_base.em(procData, lb=20/1e4)
    # Fourier transform the fid function
    procData = ng.proc_base.fft(procData)
    # Apply phase correction
    if autoPhase:
        procData = ng.proc_autophase.autops(procData, "peak_minima")
        pass
    else:
        zerothCorrection = dic["procs"]["PHC0"]
        firstCorrection = dic["procs"]["PHC1"]
        procData = ng.proc_base.ps(procData, zerothCorrection, firstCorrection)
    # Delete imaginary part
    procData = ng.proc_base.di(procData)
    # Calculate digital resolution
    digitalResolution = dic["procs"]["SW_p"] / dic["procs"]["SI"]
    # Transform into pandas DataFrame
    nmrDF = pd.DataFrame(procData, columns=["Intensity"])
    nmrDF["Frequency"] = nmrDF.index * digitalResolution
    # Apply frecuency offset
    offset = dic["procs"]["SF"]*dic["procs"]["OFFSET"]
    nmrDF["Frequency"] = nmrDF["Frequency"] - offset
    # Return the DataFrame
    return nmrDF

def filter_peaks(peaksDataFrame, freqLimit, intensityLimit):
    """
    Filter peaks obtained from the find_peaks scipy.signal function on a NMR spectrum.
    Requires the parameter:
        peaksDataFrame: DataFrame with the peaks obtained from the find_peaks function
        freqLimit: Tuple with the lower and upper frequency limits
        intensityLimit: Tuple with the lower and upper intensity limits
    Returns a DataFrame with the filtered peaks
    """
    freqFilter = (freqLimit[0] < peaksDataFrame["Frequency"]) & (
        peaksDataFrame["Frequency"] < freqLimit[1]
    )
    intensityFilter = (intensityLimit[0] < peaksDataFrame["Intensity"]) & (
        peaksDataFrame["Intensity"] < intensityLimit[1]
    )
    filteredPeaks = peaksDataFrame.loc[(freqFilter & intensityFilter)]
    return filteredPeaks


def calculate_quad_splittings(peaksDataFrame):
    """
    Calculate the quadrupolar splittings of a 2H-NMR spectrum.
    Requires the parameter:
        peaksDataFrame: DataFrame containing the peaks of the spectrum. Must have a column named "Frequency"
    Returns a list with the splittings
    """
    negativeSeries = peaksDataFrame.loc[(peaksDataFrame["Frequency"] < 0)][
        "Frequency"
    ].sort_values(ascending=False)
    positiveSeries = peaksDataFrame.loc[(peaksDataFrame["Frequency"] > 0)]["Frequency"]

    splittings = []
    for iii in range(min(len(positiveSeries), len(negativeSeries))):
        splittings.append(positiveSeries.iloc[iii] - negativeSeries.iloc[iii])

    if len(positiveSeries) == len(negativeSeries):
        return splittings
    elif len(positiveSeries) > len(negativeSeries):
        shorterSeries, longerSeries = negativeSeries, positiveSeries
    else:
        shorterSeries, longerSeries = positiveSeries, negativeSeries

    diff = len(longerSeries) - len(shorterSeries)
    for jjj in range(diff):
        splittings.append(
            shorterSeries.iloc[-1] - longerSeries.iloc[len(shorterSeries) + jjj]
        )
    return splittings
