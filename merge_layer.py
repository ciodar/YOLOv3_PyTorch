class MergeLayer(nn.Module):
    def __init__(self, use_cuda=None, layers=[]):
        super(YoloLayer, self).__init__()
        use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.layers = layers

    def get_mask_boxes(self, output):
        return {'x':output}

    def forward(self,outputs):
        if len(self.layers) == 1:
            x = outputs[self.layers[0]]
        else:
            o = [outputs[i] for i in self.layers]
            x = torch.cat(o, 1)
        return x
