

# get the output size
def down(in_size, kernel_size, times, stride=2, padding=1):
    diff = 2 * padding - kernel_size + stride
    out_size = in_size
    for i in range(times):
        out_size = (out_size + diff) // stride
    return out_size
