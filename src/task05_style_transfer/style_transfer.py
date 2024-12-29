import torch
import torch.nn as nn


class StyleTransfer:
    def __init__(self, content_weight, style_weight):
        self.content_weight = content_weight
        self.style_weight = style_weight

    def transfer(self, content_image, style_image, num_steps=1000):
        """
        Perform neural style transfer.
        """
        pass
