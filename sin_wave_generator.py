import numpy
import numpy.fft
import configparser
import matplotlib.pyplot as plt
import math
import sys
import csv
import random

# Путь до файла с параметрами
configPath = "sin_wave_generator.config"

outputPath = ["discrete_sin_wave_0.csv", "discrete_sin_wave_1.csv", "discrete_sin_wave_2.csv", "discrete_sin_wave_3.csv"]
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
    #phase_shift = config.getfloat("WaveParameters", "phase_shift")
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
        dsignal.append(amplitude * math.sin(2 * math.pi * i * freq_signal/freq_strobe + phase_shift)) #+ (random.uniform(0, 0.3) - 0.15) )
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

# def filter(dsignal):
#     #alpha = 0.99312
#     #0.99701
#     alpha = 0.994461
#     beta = 1 - alpha
#     out = list()
#     i = 0
#     while i < samples:
#         if i == 0:
#             out.append(dsignal[i]) #out[i] = dsignal[i]
#         else:
#             out.append(alpha * out[i-1] + beta * dsignal[i]) #out[i] = alpha * out[i-1] + beta * dsignal[i]
#         i = i + 1
#     return out

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

theta = float(sys.argv[1]) # принимаем параметр в градусах (шаг)
theta = numpy.deg2rad(theta) # переводим его в радианы
r = 0.01 # Distance between sound source and microphone system (1 cm)
phi = [0, math.pi/2, math.pi, 3/4*math.pi]

readParameters(configPath)

for i in range(0, len(phi)):
    phase_shift = ( -r / 340) * math.cos(theta - phi[i])
    [dsignal, time] = discreteSignal(amplitude, phase_shift, signal_bias, freq_signal, samples, freq_strobe)
    resultsFile(outputPath[i], time, dsignal)
    # plt.scatter([time],[dsignal], s=1)
    # plt.xlabel('time')
    # plt.show()


# out_signal = filter(dsignal)
# plt.scatter([time],[dsignal], s=1)
# # plt.scatter([time],[out_signal], s=1)
# plt.xlabel('time')
# plt.show()
# resultsFile(outputPath, time, dsignal)
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