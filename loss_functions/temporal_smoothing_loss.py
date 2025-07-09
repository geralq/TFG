import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha      # tensor of shape (C,)
        self.ignore = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (N, C), targets: (N,)
        N, C = logits.shape
        valid = (targets >= 0) & (targets < C)
        if valid.sum()==0:
            return torch.tensor(0., device=logits.device, requires_grad=True)

        logits = logits[valid]        # (M,C)
        t      = targets[valid]       # (M,)
        logp   = F.log_softmax(logits, dim=1)
        p      = logp.exp()

        idx    = torch.arange(logp.size(0), device=logp.device)
        logp_t = logp[idx, t]         # (M,)
        p_t    = p   [idx, t]

        focal  = (1 - p_t)**self.gamma
        if self.alpha is not None:
            a_t   = self.alpha[t]     # (M,)
            logp_t = a_t * logp_t

        loss   = - focal * logp_t     # (M,)
        if self.reduction=='mean':    return loss.mean()
        if self.reduction=='sum':     return loss.sum()
        return loss

class TemporalSmoothingLoss(nn.Module):
    def __init__(self, tau=4):
        super().__init__()
        self.tau = tau
    def forward(self, log_probs, mask=None):
        # log_probs: (B, C, T), mask: (B,T) bool
        diffs = (log_probs[:,:,1:] - log_probs[:,:,:-1]).abs()
        if mask is not None:
            vp = (mask[:,1:] & mask[:,:-1]).unsqueeze(1)
            diffs = diffs[vp.expand_as(diffs)]
            if diffs.numel()==0:
                return torch.tensor(0., device=log_probs.device)
        return torch.clamp(diffs, max=self.tau).pow(2).mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_smooth=0.15, tau=4, alpha=None, gamma=2.0):
        super().__init__()
        self.ce     = FocalLoss( gamma=gamma,
                                 alpha=alpha,
                                 ignore_index=-100,
                                 reduction='mean' )
        self.smooth = TemporalSmoothingLoss(tau)
        self.w      = weight_smooth

    def forward(self, logits, targets):
        # logits: (B,C,T), targets: (B,T)
        B,C,T = logits.shape
        flat_logits = logits.permute(0,2,1).reshape(B*T, C)
        flat_targ   = targets.reshape(-1)
        fl = self.ce(flat_logits, flat_targ)

        logp = F.log_softmax(logits, dim=1)
        mask = targets>=0
        sl = self.smooth(logp, mask=mask)
        return fl + self.w*sl
