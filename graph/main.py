import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

plt.rcParams["font.family"] = "Times New Roman"
rc('axes', titlesize=14, labelsize=12)
rc('legend', fontsize=12)

def configure_plot(ax, title, xlabel, ylabel):
    ax.set_title(title, pad=20)
    ax.set_xlabel(xlabel, labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def plot_interpolation(x, y, title, xlabel='Размер квадратных матриц', ylabel='Время вычислений (в миллисекундах)'):
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    
    ax.scatter(x, y, color='black', s=1, label='Ароксимируемые точки', alpha=0.8)
    
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, 'b-', linewidth=1.5, label=f'Аппроксимация')

    print(poly)
    
    configure_plot(ax, title, xlabel, ylabel)
    ax.legend(framealpha=1)
    fig.tight_layout(pad=2.0)
    plt.show()

def plot_standard(data):
    x = np.arange(len(data))
    
    fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.plot(x[:100], data[:100, 0], label='CPU')
    ax1.plot(x[:100], data[:100, 1], label='CPU Multithread')
    ax1.plot(x[:100], data[:100, 2], label='CUDA')
    configure_plot(ax1, 'Зависимость скорости вычислений от размера матриц', 'Размер квадратных матриц', 'Время вычислений (в миллисекундах)')
    ax1.legend()
    plt.show()
    
    fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=150)
    ax2.plot(x, data[:, 0], label='CPU')
    ax2.plot(x, data[:, 1], label='CPU Multithread')
    ax2.plot(x, data[:, 2], label='CUDA')
    configure_plot(ax2, 'Зависимость скорости вычислений от размера матриц', 'Размер квадратных матриц', 'Время вычислений (в миллисекундах)')
    ax2.legend()
    plt.show()

def plot_transport(data):
    x = np.arange(len(data))

    fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=150)
    ax3.plot(x, data, 'k-', label='CUDA')
    configure_plot(ax3, 'Зависимость скорости копирования от размера матриц', 'Размер квадратных матриц', 'Время копирования (в миллисекундах)')
    ax3.legend()
    plt.show()

data = np.loadtxt("../log/matrix.txt")
x = np.arange(len(data))

plot_standard(data)
plot_interpolation(x, data[:, 0], 'Аппроксимация времени вычислений на CPU')
plot_interpolation(x, data[:, 1], 'Аппроксимация времени вычислений на CPU Multithread')
plot_interpolation(x, data[:, 2], 'Аппроксимация времени вычислений на GPU')

data_transport = np.loadtxt("../log/transport.txt")
x_transport = np.arange(len(data_transport))
plot_transport(data_transport)
plot_interpolation(x_transport, data_transport, 'Аппроксимация времени копирования данных на GPU')

