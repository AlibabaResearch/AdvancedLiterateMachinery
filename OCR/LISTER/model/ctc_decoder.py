import torch
import torch.nn as nn


class CTCDecoder(nn.Module):
    def __init__(self, num_classes, feat_dim, blank_id):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.blank_id = blank_id
    
    def forward(self, x, mask):
        assert x.size(2) == 1, "For CTC, the height must be squeezed to 1!"
        mask = mask.flatten(1)
        mask_pad = (1 - mask).round().bool()

        x = x.permute(0, 2, 3, 1).flatten(1, 2)

        y_pred = self.fc(x) # [b, T, nc]
        # change the masked predictions to blank absolutely
        mask_pad_expd = mask_pad.unsqueeze(-1).expand_as(y_pred)
        y_pred.masked_fill_(mask_pad_expd, float('-inf'))
        mask_pad_blk = mask_pad_expd.clone()
        mask_pad_blk[:, :, :self.blank_id] = False
        mask_pad_blk[:, :, self.blank_id+1:] = False
        y_pred.masked_fill_(mask_pad_blk, 0.0)

        ret = dict(
            logits=y_pred,
            char_masks=mask,
        )
        return ret
