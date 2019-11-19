import matplotlib.pyplot as plt
import numpy as np

def main():
    t = np.linspace(0, 4*np.pi, 1000)
    fig1, ax = plt.subplots()
    ax.plot(t, np.cos(t))
    ax.plot(t, np.sin(t))

    # inception(inception(fig1))
    plt.show()

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.renderer.buffer_rgba()
    ncols, nrows = fig.canvas.get_width_height()
    print(ncols, nrows)
    return np.array(buf)[..., :3].reshape(nrows, ncols, 3)


def inception(fig):
    newfig, ax = plt.subplots()
    ax.imshow(fig2rgb_array(fig))
    return newfig

main()