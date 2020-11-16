

# get the output size
def down(
    in_size, kernel_size, stride, padding, 
    times=1, dilation=1
):
    diff = in_size + 2 * padding - dilation * (kernel_size - 1) - 1 + stride
    out_size = diff // stride
    print(out_size)
    if times <= 1:
        return out_size
    else:
        return down(
            out_size, kernel_size, stride, padding,
            times-1, dilation
        )

def up(
    in_size, kernel_size, stride, padding,
    times=1, dilation=1, out_padding=0
):
    assert isinstance(times, int), "times should be integer"
    out_size = (in_size - 1) * stride - 2 * padding + dilation * (kernel_size -1) + \
            out_padding + 1
    if times <= 1:
        return out_size
    else:
        return up(
            out_size, kernel_size, stride, padding,
            times-1, dilation, out_padding
        )

