import numpy as np


if __name__ == '__main__':
    a = np.random.randint(1, 10, size=10)
    b = np.random.randint(1, 10, size=10)
    data_arr = []

    for a_i, b_i in zip(a, b):
        data = (a_i, b_i)
        data_arr.append(data)

    # match_points = sorted(data_arr)
    match_points = sorted(data_arr, key=lambda x: x[0] / x[1])

    print('xxx')
