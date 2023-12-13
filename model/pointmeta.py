import torch
import torch.nn as nn

class BasePartSeg(BaseSeg):
    def __init__(self, encoder_args=None, decoder_args=None, cls_args=None, **kwargs):
        super().__init__(encoder_args, decoder_args, cls_args, **kwargs)

    def forward(self, data):
        if hasattr(data, 'keys'):
            q, p0, f0, cls0 = data['q'], data['pos'], data['x'], data['cls']
        else:
            if f0 is None:
                f0 = p0.transpose(1, 2).contiguous()
        p, f = self.encoder.forward_seg_feat(q, p0, f0)
  
        if self.decoder is not None:
            # print(cls0)
            f = self.decoder(p, f, cls0).squeeze(-1)
        elif isinstance(f, list):
            f = f[-1]
        if self.head is not None:
            f = self.head(f)
        return f