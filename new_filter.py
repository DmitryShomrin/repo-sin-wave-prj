import numpy
import numpy.fft
import csv
import matplotlib.pyplot as plt
import cmath
import sys
import subprocess
import time

# Constants
_b = [1, 0, 0, 0]
_phi = [0, cmath.pi / 2, cmath.pi, cmath.pi * 3 / 2]
_theta21 = cmath.pi / 1.18
_theta22 = cmath.pi / 1.69
_r = 0.01 # Distance between sound source and microphone system (1 cm)
f = 1000 # частота сигнала

# Входные значения
step = float(sys.argv[1]) # шаг в градусах от 0 до 360
step = numpy.deg2rad(step)

if len(sys.argv) != 2:
    print("Usage: python ./eval.py step(degrees)")
    sys.exit(0)

path = ["discrete_sin_wave_0.csv", "discrete_sin_wave_1.csv", "discrete_sin_wave_2.csv", "discrete_sin_wave_3.csv"]
# Функция чтения
def read_signal(path):
    reader = csv.reader(path)
    row = 1
    dsignal = list()
    for row in reader:
        dsignal.append(row[1])
    return dsignal

# Вспомогательные функции для расчета фильтра

def exp_pow(pow_val : float) -> complex:
    complex_val = complex(0, pow_val)
    return cmath.exp(complex_val)

def build_matrix(w : float, phi : list, theta21 : float, theta22 : float) -> list:
    return [
        [exp_pow(-w * cmath.cos(angle)) for angle in phi],
        [exp_pow(-w * cmath.cos(theta21 - angle)) for angle in phi],
        [exp_pow(-w * cmath.cos(theta22 - angle)) for angle in phi],
        [0, 1, 0, -1]
    ]

# Функция расчета фильтра (частота зафиксирована в 1кГц)
def calc_filter(w : float, phi: list, theta21 : float, theta22 : float, b4 : list) -> numpy.ndarray:
    A4 = build_matrix(w, phi, theta21, theta22)
    A4_INV = numpy.linalg.inv(A4)
    h_w = numpy.array(A4_INV)
    b4_arr = numpy.array(b4)
    h_w = h_w.dot(b4_arr.T)
    return h_w

def angular_frequency(freq_hz: float) -> float: # Угловая частота из обычной
    return 2 * cmath.pi * freq_hz

w = angular_frequency(f) # Перевод в угловую частоту

# Расчет коэффициентов фильтра
filter = calc_filter(w, _phi, _theta21, _theta22, _b)

# Вывод расчитанных коэффициентов фильтра
print(f'FILTER: {filter}')

pre_point = complex(0,0) # Сюда будем суммировать
points = [] # Точки для построения
theta_for_polar = []
x = []
for i in range(0, 1000, 1):
    x.append(i)

for theta in numpy.arange(0, cmath.pi * 2, step): # Цикл с заданным шагом (в радианах)
    #вызвать скрипт генератор и передать на вход theta, получить 4 файла сигнала
    subprocess.Popen(['python3.8', 'sin_wave_generator.py', str(theta)])
    time.sleep(3)
    # прочитать каждый файл
    with open(path[0], "r") as signalFile:
        dsignal0 = read_signal(signalFile)
        dsignal0.remove("signal")
    with open(path[1], "r") as signalFile:
        dsignal1 = read_signal(signalFile)
        dsignal1.remove("signal")
    with open(path[2], "r") as signalFile:
        dsignal2 = read_signal(signalFile)
        dsignal2.remove("signal")
    with open(path[3], "r") as signalFile:
        dsignal3 = read_signal(signalFile)
        dsignal3.remove("signal")    
    # быстрое преобразование Фурье для каждого из 4х сигналов
    dsignal0_fft = numpy.fft.fft(dsignal0)
    # print(dsignal0_fft)
    dsignal1_fft = numpy.fft.fft(dsignal1)
    # print(dsignal1_fft)
    dsignal2_fft = numpy.fft.fft(dsignal2)
    # print(dsignal2_fft)
    dsignal3_fft = numpy.fft.fft(dsignal3)
    plt.plot(x, dsignal0_fft.real, 'ro')
    plt.show()
    # print(dsignal3_fft)
    # первое значение из ряда Фурье перемножить с фильтром
    transform_result_0 = dsignal0_fft[21] * filter[0] # Первое значение из выхода Фурье перемножаем с фильтром
    transform_result_1 = dsignal1_fft[21] * filter[1] 
    transform_result_2 = dsignal2_fft[21] * filter[2] 
    transform_result_3 = dsignal3_fft[21] * filter[3] 
    print("transform_result_0 = ", transform_result_0)
    print("transform_result_1 = ", transform_result_1)
    print("transform_result_2 = ", transform_result_2)
    print("transform_result_3 = ", transform_result_3)
    # Сложить 4 значения
    # pre_point = pre_point + transform_result_0 + transform_result_1 + transform_result_2 + transform_result_3
    pre_point = transform_result_0 + transform_result_1 + transform_result_2 + transform_result_3
    # Сделать обратное преобразование
    point = numpy.fft.ifft([pre_point])[0]
    # point = pre_point
    # print(point)
    # получить точки
    points.append(point)
    print(len(points))
    # точка это радиус, theta - угол. Отрисовать все в полярных координатах
    theta_for_polar.append(theta)


def cart2pol(x : float, y : float) -> tuple:
    rho = numpy.sqrt(x**2 + y**2)
    # rho = x
    phi = numpy.arctan2(y, x)
    return(rho, phi)

# Перегоняем точки в полярную систему
polar_r = []
polar_th = []
for point in points:
    rad, thet = cart2pol(point.real, point.imag)
    polar_r.append(rad)
    polar_th.append(thet)
# print(theta_for_polar)
# for point in points:
#     r  = numpy.sqrt(point.real**2 + point.imag**2)
#     polar_r.append(r)
# print(polar_r)
print('grafik')
# Строим полярный график
plt.polar(theta_for_polar, polar_r, 'ro')
# plt.polar(theta_for_polar, polar_r)
plt.show()