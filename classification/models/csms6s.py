import torch
import torch.nn as nn


class Predictor(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=384, k=1):
        super().__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=1, stride=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, k, kernel_size=1, stride=1),
        )

    # self.sm =  nn.Softmax(dim=-1)

    def forward(self, x):
        mask = self.out_conv(x)
        B, C, H, W = mask.size()
        # mask_softmax = self.sm(mask.flatten(2))
        return mask


# pytorch differentiable gather =============
class gather_differentiable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, decision_idx: torch.Tensor):
        _, C, _ = x.shape
        B, K, L = decision_idx.shape
        decision_idx_bkcl = decision_idx.unsqueeze(2).repeat(1, 1, C, 1)
        ctx.saved_index = decision_idx_bkcl
        x_sequence = x.unsqueeze(1).repeat(1, K, 1, 1)
        xs = torch.gather(input=x_sequence, dim=-1, index=decision_idx_bkcl)  # [b, k, c, hw]
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        y_sequence = ys.scatter(dim=-1, index=ctx.saved_index, src=ys)
        y = y_sequence.sum(dim=1)
        # index_bkl = ctx.saved_index.mean(dim=2)
        index_bkl = ctx.saved_index[:, :, 0, :]
        return y, index_bkl


# pytorch differentiable scatter =============
class scatter_differentiable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, decision_idx: torch.Tensor):
        _, K, C, _ = x.shape
        B, _, L = decision_idx.shape
        decision_idx_bkcl = decision_idx.unsqueeze(2).repeat(1, 1, C, 1)
        ctx.saved_index = decision_idx_bkcl
        x_sequence = x.scatter(dim=-1, index=decision_idx_bkcl, src=x)
        xs = x_sequence.sum(dim=1)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, K, C, L = ctx.saved_index.shape
        decision_idx_bkcl = ctx.saved_index
        y_sequence = ys.unsqueeze(1).repeat(1, K, 1, 1)
        y = torch.gather(input=y_sequence, dim=-1, index=decision_idx_bkcl)  # [b, k, c, hw]
        index_bkl = ctx.saved_index[:, :, 0, :]

        return y, index_bkl


def channel_shuffle_divide(x):
    b, c_g, groups, h, w = x.shape
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    x = x.view(b, c_g, -1, h, w)

    return x


def channel_shuffle(x, groups):
    b, c, h, w = x.shape
    c_g = c // groups

    x = x.view(b, groups, c_g, h, w)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(b, -1, h, w)
    return x


# these are for ablations =============
class Scan_FB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        ctx.shape = (B, C, L)
        xs = x.new_empty((B, 2, C, L))

        xs[:, 0] = x
        xs[:, 1] = x.flip(-1)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, L = ctx.shape
        y = ys[:, 0, :, :] + ys[:, 1, :, :].flip(-1)
        return y.view(B, C, L).contiguous()


class Merge_FB(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, L = ys.shape
        ctx.shape = (B, K, C, L)
        y = ys[:, 0, :, :] + ys[:, 1, :, :].flip(-1)
        return y.contiguous()

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, K, C, L = ctx.shape
        xs = x.new_empty((B, K, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.flip(-1)
        return xs



# pytorch cross scan =============
class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


# pytorch cross scan =============
class CrossScan_3D_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, Cz, Z, H, W = x.shape
        ctx.shape = (B, Cz, Z, H, W)
        xs = x.new_empty((B, 4, Cz, Z * H * W))
        ##########  HoW
        x1 = x.permute(0, 1, 2, 3, 4)
        xs[:, 0] = x1.flatten(2)
        xs[:, 1] = x1.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, Cz, Z, H, W = ctx.shape
        L = H * W * Z

        ##########  HoW
        ys1 = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)
        y = y1
        return y.view(B, -1, Z, H, W)


class CrossMerge_3D_2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, Cz, Z, H, W = ys.shape
        ctx.shape = (B, K, Cz, Z, H, W)
        ys = ys.view(B, K, Cz, -1)
        L = H * W * Z

        ##########  HoW
        ys1 = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)

        y = y1

        return y.view(B, -1, Z, H, W)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        B, K, Cz, Z, H, W = ctx.shape
        xs = x.new_empty((B, 4, Cz, Z * H * W))
        ##########  HoW
        x1 = x.permute(0, 1, 2, 3, 4)
        xs[:, 0] = x1.flatten(2)
        xs[:, 1] = x1.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        xs = xs.view(B, K, Cz, Z, H, W)

        return xs


# pytorch cross scan =============
class CrossScan_3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, Cz, Z, H, W = x.shape
        ctx.shape = (B, Cz, Z, H, W)
        xs = x.new_empty((B, 12, Cz, Z * H * W))
        ##########  HoW
        x1 = x.permute(0, 1, 2, 3, 4)
        xs[:, 0] = x1.flatten(2)
        xs[:, 1] = x1.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        ########## HoZ
        x2 = x.permute(0, 1, 4, 3, 2)
        xs[:, 4] = x2.flatten(2)
        xs[:, 5] = x2.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        ########## WoZ
        x3 = x.permute(0, 1, 3, 4, 2)
        xs[:, 8] = x3.flatten(2)
        xs[:, 9] = x3.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 10:12] = torch.flip(xs[:, 8:10], dims=[-1])

        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, Cz, Z, H, W = ctx.shape
        L = H * W * Z

        ##########  HoW
        ys1 = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)
        ########## HoZ
        ys2 = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y2 = ys2[:, 0] + ys2[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)
        ########## WoZ
        ys3 = ys[:, 8:10] + ys[:, 10:12].flip(dims=[-1]).view(B, 2, -1, L)
        y3 = ys3[:, 0] + ys3[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)

        y = y1 + y2 + y3
        return y.view(B, -1, Z, H, W)


class CrossMerge_3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, Cz, Z, H, W = ys.shape
        ctx.shape = (B, K, Cz, Z, H, W)
        ys = ys.view(B, K, Cz, -1)
        L = H * W * Z

        ##########  HoW
        ys1 = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y1 = ys1[:, 0] + ys1[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)
        ########## HoZ
        ys2 = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        y2 = ys2[:, 0] + ys2[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)
        ########## WoZ
        ys3 = ys[:, 8:10] + ys[:, 10:12].flip(dims=[-1]).view(B, 2, -1, L)
        y3 = ys3[:, 0] + ys3[:, 1].view(B, -1, Z, W, H).transpose(dim0=3, dim1=4).contiguous().view(B, -1, L)

        y = y1 + y2 + y3

        return y.view(B, -1, Z, H, W)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        B, K, Cz, Z, H, W = ctx.shape
        xs = x.new_empty((B, 12, Cz, Z * H * W))
        ##########  HoW
        x1 = x.permute(0, 1, 2, 3, 4)
        xs[:, 0] = x1.flatten(2)
        xs[:, 1] = x1.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])

        ########## HoZ
        x2 = x.permute(0, 1, 4, 3, 2)
        xs[:, 4] = x2.flatten(2)
        xs[:, 5] = x2.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        ########## WoZ
        x3 = x.permute(0, 1, 3, 4, 2)
        xs[:, 8] = x3.flatten(2)
        xs[:, 9] = x3.transpose(dim0=3, dim1=4).flatten(2)
        xs[:, 10:12] = torch.flip(xs[:, 8:10], dims=[-1])

        xs = xs.view(B, K, Cz, Z, H, W)

        return xs


# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        return ys.sum(1).view(B, -1, H, W)


class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        return ys.contiguous().sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, L = x.shape
        x = x.view(B, 1, C, H * W).repeat(1, 2, 1, 1)
        x = torch.cat([x, x.flip(dims=[-1])], dim=1)
        return x.view(B, 4, C, H, W)


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        x = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1)
        return x

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        return ys.view(B, 4, -1, H, W).sum(1)


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, C, H, W = ys.shape
        ctx.shape = (B, C, H, W)
        return ys.view(B, 4, -1, H * W).sum(1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        B, C, H, W = ctx.shape
        return x.view(B, 1, C, H, W).repeat(1, 4, 1, 1, 1)


# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    # print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    # print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    # print(e, flush=True)


def check_nan_inf(tag: str, x: torch.Tensor, enable=True):
    if enable:
        if torch.isinf(x).any() or torch.isnan(x).any():
            print(tag, torch.isinf(x).any(), torch.isnan(x).any(), flush=True)
            import pdb;
            pdb.set_trace()


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    assert not with_complex
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


# this is only for selective_scan_ref...
def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try:
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
# comment all checks if inside cross_selective_scan
class SelectiveScanMamba(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1,
                oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops




