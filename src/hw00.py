import numpy as np
import matplotlib.pyplot as plt
import os


def main():
    print("Hello World!")

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure()
    plt.plot(x, y)

    save_path = os.path.join(os.path.dirname(__file__), "../outputs/figures/s02.png")
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()
