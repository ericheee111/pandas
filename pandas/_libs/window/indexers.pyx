# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np

from numpy cimport (
    int64_t,
    ndarray,
)

# Cython routines for window indexers


cdef void _calculate_variable_window_bounds_group(
    int64_t window_size,
    bint center,
    bint left_closed,
    bint right_closed,
    const int64_t* index,
    Py_ssize_t index_stride,
    int64_t group_start,
    int64_t group_end,
    int64_t* start,
    int64_t* end,
) noexcept nogil:
    cdef:
        int64_t start_bound, end_bound, index_growth_sign = 1
        Py_ssize_t i, j

    if group_end <= group_start:
        return

    if index[(group_end - 1) * index_stride] < index[group_start * index_stride]:
        index_growth_sign = -1

    start[group_start] = group_start
    if right_closed:
        end[group_start] = group_start + 1
    else:
        end[group_start] = group_start

    if center:
        end_bound = (
            index[group_start * index_stride]
            + index_growth_sign * window_size / 2
        )
        for j in range(group_start, group_end):
            if (index[j * index_stride] - end_bound) * index_growth_sign < 0:
                end[group_start] = j + 1
            elif (
                (index[j * index_stride] - end_bound) * index_growth_sign == 0
                and right_closed
            ):
                end[group_start] = j + 1
            elif (index[j * index_stride] - end_bound) * index_growth_sign >= 0:
                end[group_start] = j
                break

    for i in range(group_start + 1, group_end):
        if center:
            end_bound = (
                index[i * index_stride]
                + index_growth_sign * window_size / 2
            )
            start_bound = (
                index[i * index_stride]
                - index_growth_sign * window_size / 2
            )
        else:
            end_bound = index[i * index_stride]
            start_bound = (
                index[i * index_stride] - index_growth_sign * window_size
            )

        if left_closed:
            start_bound -= index_growth_sign

        start[i] = i
        for j in range(start[i - 1], i):
            if (index[j * index_stride] - start_bound) * index_growth_sign > 0:
                start[i] = j
                break

        if center:
            for j in range(end[i - 1], group_end + 1):
                if j == group_end:
                    end[i] = j
                elif (
                    (index[j * index_stride] - end_bound) * index_growth_sign == 0
                    and right_closed
                ):
                    end[i] = j + 1
                elif (
                    (index[j * index_stride] - end_bound) * index_growth_sign
                    >= 0
                ):
                    end[i] = j
                    break
        elif (
            index[end[i - 1] * index_stride] == end_bound
            and not right_closed
        ):
            end[i] = end[i - 1] + 1
        elif (
            (index[end[i - 1] * index_stride] - end_bound)
            * index_growth_sign
            <= 0
        ):
            end[i] = i + 1
        else:
            end[i] = end[i - 1]

        if not right_closed and not center:
            end[i] -= 1


def calculate_variable_window_bounds(
    int64_t num_values,
    int64_t window_size,
    object min_periods,  # unused but here to match get_window_bounds signature
    bint center,
    str closed,
    const int64_t[:] index
):
    """
    Calculate window boundaries for rolling windows from a time offset.

    Parameters
    ----------
    num_values : int64
        total number of values

    window_size : int64
        window size calculated from the offset

    min_periods : object
        ignored, exists for compatibility

    center : bint
        center the rolling window on the current observation

    closed : str
        string of side of the window that should be closed

    index : ndarray[int64]
        time series index to roll over

    Returns
    -------
    (ndarray[int64], ndarray[int64])
    """
    cdef:
        bint left_closed = False
        bint right_closed = False
        ndarray[int64_t, ndim=1] start, end
        Py_ssize_t index_stride = index.strides[0] // sizeof(int64_t)

    if num_values <= 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    # default is 'right'
    if closed is None:
        closed = "right"

    if closed in ["right", "both"]:
        right_closed = True

    if closed in ["left", "both"]:
        left_closed = True

    # GH 43997:
    # If the forward and the backward facing windows
    # would result in a fraction of 1/2 a nanosecond
    # we need to make both interval ends inclusive.
    if center and window_size % 2 == 1:
        right_closed = True
        left_closed = True

    start = np.empty(num_values, dtype="int64")
    end = np.empty(num_values, dtype="int64")

    with nogil:
        _calculate_variable_window_bounds_group(
            window_size,
            center,
            left_closed,
            right_closed,
            &index[0],
            index_stride,
            0,
            num_values,
            &start[0],
            &end[0],
        )

    return start, end


def calculate_variable_window_bounds_grouped(
    int64_t num_values,
    int64_t window_size,
    object min_periods,
    bint center,
    str closed,
    ndarray[int64_t, ndim=1] index,
    ndarray[int64_t, ndim=1] group_starts,
):
    """
    Calculate time-based rolling-window boundaries for contiguous groups.
    """
    cdef:
        bint left_closed = False
        bint right_closed = False
        ndarray[int64_t, ndim=1] start, end
        Py_ssize_t group, num_groups = len(group_starts) - 1
        Py_ssize_t index_stride = index.strides[0] // sizeof(int64_t)

    if num_values <= 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

    if closed is None:
        closed = "right"

    if closed in ["right", "both"]:
        right_closed = True

    if closed in ["left", "both"]:
        left_closed = True

    if center and window_size % 2 == 1:
        right_closed = True
        left_closed = True

    start = np.empty(num_values, dtype="int64")
    end = np.empty(num_values, dtype="int64")

    with nogil:
        for group in range(num_groups):
            _calculate_variable_window_bounds_group(
                window_size,
                center,
                left_closed,
                right_closed,
                &index[0],
                index_stride,
                group_starts[group],
                group_starts[group + 1],
                &start[0],
                &end[0],
            )

    return start, end
