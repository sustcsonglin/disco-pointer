def stripe(x, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)
    else:
        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,

                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)



def stripe_add_(x, y, n, w, offset=(0, 0), dim=1):
    x, seq_len = x.contiguous(), x.size(2)
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (seq_len + 1) * numel
    stride[2] = (1 if dim == 1 else seq_len) * numel
    if len(x.shape) > 3:
        a = x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

        return x.as_strided(size=(x.shape[0], n, w, *list(x.shape[3:])),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel).copy_(a + y)
    else:
        a = x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel)

        return x.as_strided(size=(x.shape[0], n, w),
                            stride=stride,
                            storage_offset=(offset[0] * seq_len + offset[1]) * numel).copy_(y + a)




def diagonal_copy_(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(y)
    else:
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(y)


def diagonal_add_(x, y, w):
    # size of x: (batch, N, N, nt)
    # size of y: (batch, N, nt)
    # the function aims to copy y to the diagonal of x (dim1 and dim2) without any copy of tensor.
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        a =  x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     )
        x.as_strided(size=(x.shape[0], seq_len - w,  *list(x.shape[3:])),
                     stride=new_stride,
                     storage_offset= w * stride[2]
                     ).copy_(a + y)
    else:
        a =  x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     )
        x.as_strided(size=(x.shape[0], seq_len - w),
                     stride=new_stride,
                     storage_offset=w * stride[2]
                     ).copy_(a + y)

def diagonal(x, w):
    x, seq_len = x.contiguous(), x.size(1)
    stride, numel = list(x.stride()), x[:, 0, 0].numel()
    new_stride = []
    new_stride.append(stride[0])
    new_stride.append(stride[1] + stride[2])
    if len(x.shape) > 3:
        new_stride.extend(stride[3:])
        return x.as_strided(size=(x.shape[0], seq_len - w, *list(x.shape[3:])),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )
    else:
        return x.as_strided(size=(x.shape[0], seq_len - w),
                            stride=new_stride,
                            storage_offset=w * stride[2]
                            )

def gradient_of_softmax(a, g_out):
    return a * (g_out - (g_out*a).sum(-1).unsqueeze(-1))
