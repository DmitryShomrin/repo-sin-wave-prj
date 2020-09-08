import numpy
import numpy.fft
import csv
import matplotlib.pyplot as plt
import cmath
import sys

path = "discrete_sin_wave.csv"

def read_signal(path):
    reader = csv.reader(path)
    row = 1
    dsignal = list()
    for row in reader:
        dsignal.append(row[1])
    return dsignal

'''
Функции для расчета фильтра
'''

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

def calc_filter(w : float, phi: list, theta21 : float, theta22 : float, b4 : list) -> numpy.ndarray:
    A4 = build_matrix(w, phi, theta21, theta22)
    A4_INV = numpy.linalg.inv(A4)
    h_w = numpy.array(A4_INV)
    b4_arr = numpy.array(b4)
    h_w = h_w.dot(b4_arr.T)
    return h_w



'''
Функции для построения диаграммы направленности
'''

def angular_frequency(freq_hz: float) -> float: # Угловая частота из обычной
    return 2 * cmath.pi * freq_hz

def tau(r : float, theta: float, phi : list, m : int) -> float:
    return (- r / 340) * cmath.cos(theta - phi[m])

def phase_shift(_tau: float, _w : float) -> float:
    return cmath.sin(_tau * _w)

def direct_transform_for_mic(value: float, dsignal) -> complex:
    value = numpy.array(value, dtype=numpy.complex64)
    dsignal = numpy.array(dsignal, dtype=numpy.float64)
    value1 = value * dsignal
    return numpy.fft.fft(value1)
    # return numpy.fft.fft([value] * 500 + [0] * 12)

# def direct_transform_for_mic(value: float) -> complex:

#     return numpy.fft.fft([value] * 500)

def cart2pol(x : float, y : float) -> tuple:
    rho = numpy.sqrt(x**2 + y**2)
    phi = numpy.arctan2(y, x)
    return(rho, phi)


# Constants
_b = [1, 0, 0, 0]
_phi = [0, cmath.pi / 2, cmath.pi, cmath.pi * 3 / 2]
_theta21 = cmath.pi / 1.18
_theta22 = cmath.pi / 1.69
_r = 0.01 # Distance between sound source and microphone system (1 cm)

if len(sys.argv) != 3:
    print("Usage: python ./eval.py frequency(Hz) step(degrees)")
    sys.exit(0)

with open(path, "r") as signalFile:
    dsignal = read_signal(signalFile)
    dsignal.remove("signal")

# Входные аргументы
f    = float(sys.argv[1])    # Первый аргумент, частота в герцах 
step = float(sys.argv[2])    # Второй аргумент, шаг изменения угла (в градусах)
step = numpy.deg2rad(step)

w = angular_frequency(f)    # Угловая частота

_h_filter = calc_filter(w, _phi, _theta21, _theta22, _b) # Фильтр от введенной частоты

print(f'FILTER: {_h_filter}')

points = [] # Точки для построения

for angle in numpy.arange(0, cmath.pi * 2, step): # Цикл с заданным шагом (в радианах)
    
    pre_point = complex(0,0) # Сюда будем суммировать
    
    for mic in range(0,4):
        _tau = tau(_r, angle, _phi, mic)        # Считаем задержку
        _shift = phase_shift(w, _tau)           # Считаем сдвиг фазы 
        #print(_shift)

        # print(f'SHIFT: {_shift}')

        transform_result = direct_transform_for_mic(_shift, dsignal) # Значения сдвига фазы в виде 500 точек кладем в Фурье
        #print(transform_result.size)

        transform_result = transform_result[0] * _h_filter[mic] # Первое значение из выхода Фурье перемножаем с фильтром

        pre_point = pre_point + transform_result    # Складываем значения для текущего микрофона с накопленным
    
    point = numpy.fft.ifft([pre_point])[0] # Обратное преобразование
    points.append(point)

#print(points)
# Перегоняем точки в полярную систему
polar_r = []
polar_th = []
for point in points:
    rad, thet = cart2pol(point.real, point.imag)
    polar_r.append(rad)
    polar_th.append(thet)

# Строим полярный график
plt.polar(polar_r, polar_th)
plt.show()

