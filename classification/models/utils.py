
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_expansion(x, H, W):
  # x [b, 1, 4, 1, 1]
    b, _, num_win, _, _ = x.shape
    H1, W1 = int(H/2), int(W/2)
    num_win1 = int(num_win/2)

    x = x.reshape(b, 1, num_win1, num_win1, 1).squeeze(-1)
    x = F.interpolate(x, scale_factor=H1)
    """
    x = x.repeat(1, 1, 1, H1, W1)
    x = x.reshape(b, 1, num_win1, num_win1, H1, W1).permute(0, 1, 2, 4, 3, 5)
    x = x.reshape(b, 1, -1).contiguous()
    """
    x_rs = x.reshape(b, 1, -1)
    return x_rs


def window_partition(x, quad_size=2):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, window_size, window_size, C)
    """
    B, C, H, W = x.shape
    H_quad = H // quad_size
    W_quad = W // quad_size

    x = x.view(B, C, quad_size, H_quad, quad_size, W_quad)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, -1, H_quad, W_quad, C)  #.permute(0, 2, 1, 3, 4)
    return windows


def window_reverse(windows):
    """
    Args:
        windows: (B, C, num_windows, window_size, window_size)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, N, H, W, C)
    """
    B, N, H_l, W_l, C = windows.shape
    scale = int((N)**0.5)
    H = H_l * scale

    W = W_l * scale

    x = windows.permute(0, 4, 1, 2, 3)
    x = x.view(B, C, N // scale, N // scale, H_l, W_l)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return x


def shift_size_generate(index_block=0, H=0):
    sz = int(H // 8)
    index = index_block //2
    if (index%4)==0:
        shift_size = (sz, sz)
        reverse_size = (-sz, -sz)
    elif (index%4)==1:
        shift_size = (-sz, -sz)
        reverse_size = (sz, sz)
    elif (index % 4) == 2:
        shift_size = (sz, -sz)
        reverse_size = (-sz, sz)
    else:
        shift_size = (-sz, sz)
        reverse_size = (sz, -sz)
    return shift_size, reverse_size


class Predictor(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        if len(x.shape) == 4:
            B, C, H, W = x.size()
            x_rs = x.reshape(B, C, -1).permute(0, 2, 1)
        else:
            B, N, C = x.size()
            H = int(N**0.5)
            x_rs = x
        x_rs = self.in_conv(x_rs)
        B, N, C = x_rs.size()

        window_scale = int(H//2)
        local_x = x_rs[:, :, :C // 2]
        global_x = x_rs[:, :, C // 2:].view(B, H, -1, C // 2).permute(0, 3, 1, 2)
        global_x_avg = F.adaptive_avg_pool2d(global_x,  (2, 2)) # [b, c, 2, 2]
        global_x_avg_concat = F.interpolate(global_x_avg, scale_factor=window_scale)
        global_x_avg_concat = global_x_avg_concat.view(B, C // 2, -1).permute(0, 2, 1).contiguous()

        x_rs = torch.cat([local_x, global_x_avg_concat], dim=-1)

        x_score = self.out_conv(x_rs)
        x_score_rs = x_score.permute(0, 2, 1).reshape(B, 2, H, -1)
        return x_score_rs



"""PyTorch code for local scan and local reverse"""


def local_scan(x, w=7, H=14, W=14, h_scan=False):
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if h_scan:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 3, 1, 4, 2, 5).reshape(B, -1, C)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, C)
    return x



def local_reverse(x, w=7, H=14, W=14, h_scan=False):
    B, L, C = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        if h_scan:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, Hg * w, Wg * w, C)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, Hg * w, Wg * w, C)
        x = x[:, :H, :W].reshape(B, -1, C)
    else:
        if h_scan:
            x = x.view(B, Wg, Hg, w, w, C).permute(0, 2, 4, 1, 3, 5).reshape(B, L, C)
        else:
            x = x.view(B, Hg, Wg, w, w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, L, C)
    return x


def local_scan_quad(x, quad=2, H=None, W=None, h_scan=False):
    B, C, L = x.shape
    x = x.view(B, C, H, W)

    quad1 = quad2 = quad
    h = math.ceil(H / quad)
    w = math.ceil(W / quad)

    if h_scan:
        x = None
    else:
        x = x.view(B, C, quad1, h, quad2, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)  # B, Hg, Wg, w, w, C

    return x

def local_scan_quad_quad(x, quad=2, H=None, W=None, h_scan=False):
    B, C, L = x.shape
    x = x.view(B, C, H, W)

    quad1 = quad2 = quad3 = quad4 = quad
    h = math.ceil((H / quad) / quad)
    w = math.ceil((W / quad) / quad)

    if h_scan:
        x = None
    else:
        x = x.view(B, C, quad1, quad3 * h, quad2, quad4 * w).view(B, C, quad1, quad3, h,  quad2, quad4, w)\
            .permute(0, 1, 2, 3, 5, 6, 4, 7).reshape(B, C, -1)  # B, Hg, Wg, w, w, C

    return x


def local_reverse_quad(y, quad=2, H=None, W=None, h_scan=False):
    B, C, L = y.shape

    quad1 = quad2 = quad
    h = math.ceil(H / quad)
    w = math.ceil(W / quad)

    if h_scan:
        y = None
    else:
        y = y.view(B, C, quad1, quad2, h, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1)

    return y


def local_reverse_quad_quad(y, quad=2,  H=None, W=None, h_scan=False):
    B, C, L = y.shape

    quad1 = quad2 = quad3 = quad4 = quad
    h = math.ceil((H / quad) / quad)
    w = math.ceil((W / quad) / quad)

    if h_scan:
        y = None
    else:
        y1 = y.view(B, C, quad1, quad3,  quad2, quad4, h, w)
        y1 = y1.permute(0, 1, 2, 3, 6, 4, 5, 7)
        y = y1.reshape(B, C, -1)

    return y




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



