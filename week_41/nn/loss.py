from .module import Module
 
class MSELoss(Module):
    def __init__(self):
      pass

    def forward(self, predictions, labels):
        assert labels.shape == predictions.shape, \
            "Labels and predictions shape does not match: {} and {}".format(labels.shape, predictions.shape)
        
        return ((predictions - labels) ** 2).sum() / predictions.numel

    def __call__(self, *inputs):
        return self.forward(*inputs)