# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from opt_einsum import contract


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
    in which :math:`x` and :math:`y` can be concatenated with bias terms.

    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.

    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.

    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """


    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        
        # [batch_size,  seq_len, seq_len, out]
        s = contract('bxi,oij,byj->bxyo', x, self.weight, y, backend='torch')
        # remove dim 1 if n_out == 1
        s = s.squeeze(-1)
        return s

    def forward_bmnx_bay_bamn(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        # [batch_size,  seq_len, seq_len, out]
        s = contract('bmni,oij,baj->bamno', x, self.weight, y, backend='torch')
        # remove dim 1 if n_out == 1
        s = s.squeeze(-1)
        return s

    def forward_blx_bay_2_bal(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = contract('blx, oxy, bay -> bal', x, self.weight, y, backend='torch')
        return s

    def forward_bx_by_2_b(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        s = contract('bx, oxy, by->bo', x, self.weight, y, backend='torch')
        return s

    def forward2(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        s = contract('ai, oij, aj -> ao', x, self.weight, y, backend='torch')
        return s


    def forward_v2(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        s = contract('ash, ac, dhc -> asd', x, y, self.weight, backend='torch').squeeze(-1)
        return s

    def forward_v3(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        s = contract('blh, bac, dhc -> bal', x, y, self.weight,  backend='torch')
        return s

class Triaffine(nn.Module):
    r"""
    Triaffine layer for second-order scoring.

    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y, z)` of the vector triple :math:`(x, y, z)` is computed as :math:`x^T z^T W y`.
    Usually, :math:`x` and :math:`y` can be concatenated with bias terms.

    References:
        - Yu Zhang, Zhenghua Li and Min Zhang. 2020.
          `Efficient Second-Order TreeCRF for Neural Dependency Parsing`_.
        - Xinyu Wang, Jingxian Huang, and Kewei Tu. 2019.
          `Second-Order Semantic Dependency Parsing with End-to-End Neural Networks`_.

    Args:
        n_in (int):
            The size of the input feature.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``False``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``False``.

    .. _Efficient Second-Order TreeCRF for Neural Dependency Parsing:
        https://www.aclweb.org/anthology/2020.acl-main.302/
    .. _Second-Order Semantic Dependency Parsing with End-to-End Neural Networks:
        https://www.aclweb.org/anthology/P19-1454/
    """

    def __init__(self, n_in, bias_x=False, bias_y=False):
        super().__init__()

        self.n_in = n_in
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_in+bias_x, n_in, n_in+bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y, z):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            z (torch.Tensor): ``[batch_size, seq_len, n_in]``.

        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, seq_len, seq_len, seq_len]``.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)

        w = contract('bzk,ikj->bzij', z, self.weight, backend='torch')
        # [batch_size, seq_len, seq_len, seq_len]
        s = contract('bxi,bzij,byj->bzxy', x, w, y, backend='torch')
        return s

    def forward_bmx_bny_baz_2_bamn(self, x, y, z):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = contract('baz,xyz->baxy', z, self.weight, backend='torch')
        s = contract('blx,baxy->baly', x, w,  backend='torch')
        s = contract('baly,bny-> baln', s, y, backend='torch')
        return s

    def forward_blx_bay_baz_2_bal(self, x, y, z):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        w = contract('baz,xyz->baxy', z, self.weight, backend='torch')
        s = contract('blx, baxy ->baly', x, w,  backend='torch')
        s = contract('baly, bay -> bal', s, y, backend='torch')
        return s

