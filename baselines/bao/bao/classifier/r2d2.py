from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BASE


class R2D2(BASE):
    '''
        META-LEARNING WITH DIFFERENTIABLE CLOSED-FORM SOLVERS
    '''

    def __init__(self, ebd_dim, args):
        super(R2D2, self).__init__(args)
        self.ebd_dim = ebd_dim

        # meta parameters to learn
        self.lam = nn.Parameter(torch.tensor(-1, dtype=torch.float))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float))
        # lambda and alpha is learned in the log space

        self._set_I_support_way()

    def _set_I_support_way(self):
        # cached tensor for speed
        self.I_support = nn.Parameter(
            torch.eye(self.args.shot * self.args.way, dtype=torch.float),
            requires_grad=False)
        self.I_way = nn.Parameter(torch.eye(self.args.way, dtype=torch.float),
                                  requires_grad=False)

    def set_way_shot(self, way: Optional[int] = None, shot: Optional[int] = None):
        recompute = False
        if way is not None and way != self.args.way:
            self.args.way = way
            recompute = True
        if shot is not None and shot != self.args.shot:
            self.args.shot = shot
            recompute = True
        if recompute:
            self._set_I_support_way()

    def _compute_w(self, XS, YS_onehot):
        '''
            Compute the W matrix of ridge regression
            @param XS: support_size x ebd_dim
            @param YS_onehot: support_size x way

            @return W: ebd_dim * way
        '''

        W = XS.t() @ torch.inverse(
            XS @ XS.t() + (10. ** self.lam) * self.I_support) @ YS_onehot

        return W

    def _label2onehot(self, Y):
        '''
            Map the labels into 0,..., way
            @param Y: batch_size

            @return Y_onehot: batch_size * ways
        '''
        Y_onehot = F.embedding(Y, self.I_way)

        return Y_onehot

    def forward(self, XS, YS, XQ, YQ=None, return_pred=False):
        '''
            @param XS (support x): support_size x ebd_dim
            @param YS (support y): support_size
            @param XQ (support x): query_size x ebd_dim
            @param YQ (support y): query_size

            @return acc
            @return loss
        '''
        if YQ is not None:
            YS, YQ = self.reidx_y(YS, YQ)
        else:
            YS = self.reidx_y(YS)

        YS_onehot = self._label2onehot(YS)

        W = self._compute_w(XS, YS_onehot)

        pred = (10.0 ** self.alpha) * XQ @ W + self.beta

        if YQ is not None:
            loss = F.cross_entropy(pred, YQ)

            acc = BASE.compute_acc(pred, YQ)
        else:
            loss = None

            acc = None

        if return_pred:
            return acc, loss, pred
        else:
            return acc, loss
