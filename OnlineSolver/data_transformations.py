import numpy as np


def meaner(array, window=20):
    size = array.shape[0]
    return np.hstack([np.mean(array[i:i + window]) for i in range(0, size, window)])


def scalling(array):
    array = array.copy()
    min_ = np.min(array)
    array -= min_
    max_ = np.max(array)
    array /= max_
    return array


def smoothing(row, win=5):
    """
    Smooth given row in window with width = win.
    """
    array = np.array(row).ravel()
    new_array = np.empty(array.shape)
    offset1 = win // 2
    offset2 = win - offset1
    array_size = len(array)
    for i in range(array_size):
        if i < offset1:
            new_array[i] = np.mean(array[:i + offset2])
        elif i > array_size - offset2:
            new_array[i] = np.mean(array[i - offset1:])
        else:
            new_array[i] = np.mean(array[i - offset1:i + offset2])
    return new_array


def reduce_point_number(array: np.array, window: int, shift: int = None) -> np.array:
    """This function reduce the number of points in the scaling maner.
    Cause of that, needed information may be lost.
    Input:      array : 1-D np.array to convert
                window  : int
                shift : int
    Returns:            1-D np.array

    Comment: shape = window * steps"""
    data = array.copy()
    data = data.ravel()
    if shift:
        if shift >= window:
            raise Exception("Shift must be less than window")
    else:
        shift = 0
    steps = data.shape[0] // window  # aka number of points to return
    to_return = np.empty((steps,))
    for i in range(steps):
        to_return[i] = data[i * window + shift]
    return to_return
