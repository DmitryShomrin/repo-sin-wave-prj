import numpy
import numpy.fft
import configparser
import matplotlib.pyplot as plt
import math
import sys
import csv

# Путь до файла с параметрами
configPath = "sin_wave_generator.config"

outputPath = "discrete_sin_wave.csv"
# Параметры описываются в секции [WaveParameters]
def readParameters(configPath):
    config = configparser.ConfigParser()
    config.read(configPath)
    global amplitude
    global samples
    global phase_shift
    global signal_bias
    global freq_signal
    global freq_strobe
    amplitude = config.getfloat("WaveParameters", "amplitude")
    phase_shift = config.getfloat("WaveParameters", "phase_shift")
    signal_bias = config.getfloat("WaveParameters", "signal_bias")
    freq_signal = config.getfloat("WaveParameters", "freq_signal")
    samples = config.getint("StrobeParameters", "samples")
    freq_strobe = config.getfloat("StrobeParameters", "freq_strobe")

# amplitude - амплитуда сигнала
# samples - number of time samples per sine wave period
# phase_shift - is the offset (phase shift) of the signal - сдвиг фазы
# signal_bias - is the signal bias - сдвиг(смещение) сигнала
# freq_signal - частота полезного сигнала
# freq_strobe - частота стробирующего сигнала

def discreteSignal(amplitude :  float, phase_shift : float, signal_bias: float, freq_signal: float, samples : int, freq_strobe: float):
    dsignal = list()
    time = list()
    i = 0
    while i < samples:
        dsignal.append(amplitude * math.sin(2 * math.pi * i * freq_signal/freq_strobe + phase_shift) )
        time.append(i/freq_strobe)
        i = i + 1
    return dsignal,time

def resultsFile(outputPath, time, dsignal):
    with open(outputPath, "w+") as output_csv:
        writer = csv.writer(output_csv, delimiter=',')
        writer.writerow(["time","signal"])
        for i in range(0, samples):
            writer.writerow([time[i],dsignal[i]])
        print("Writing results complete")

# def signal(amplitude :  float, samples : int, phase_shift : float, signal_bias: float): #-> numpy.ndarray:
#     signal1 = list()
#     for i in range(0, samples):
#         signal1.append(amplitude * math.sin(2 * math.pi * ( i + phase_shift ) / samples ) + signal_bias)
#     return signal1

# def plotSignal(samples : int):
#     x = []
#     for i in range(0, samples):
#         x.append(math.pi * i / samples)
#     plt.scatter([x],[a])
#     plt.show()

readParameters(configPath)
[dsignal, time] = discreteSignal(amplitude, phase_shift, signal_bias, freq_signal, samples, freq_strobe)
plt.scatter([time],[dsignal], s=1)
plt.xlabel('time')
plt.show()
resultsFile(outputPath, time, dsignal)
# a = signal(amplitude, samples, phase_shift, signal_bias)
# plotSignal(samples)







# time = 1
# i = 0
# x = []
# y = []
# m = []
# index = 0
# step = 0
# count = 0
# while i < time: 
#     y.append(amplitude * math.sin(1000 * i))
#     if (y[len(y)-1] > 0.999):
#         count = count + 1
#     x.append(i)
#     i = i + time/1000
    
# while step < 1000:
#     m.append(0.5 * math.sin(48000 * step))
#     step = step + 1

# plt.scatter([x],[y])
# print(count)
# # plt.scatter([x], [m], color='red', s=1)
# plt.show()
# print(a)

# plt.plot([range(0, 1000)],[[a.real for a in a]])
# plt.show()