from torch.autograd import Function

class GradientScaleLayer(Function):

    @staticmethod
    def forward(ctx, x, _lambda):
        ctx._lambda = _lambda
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx._lambda
        return grad_output, None

def gradient_scale_layer(x, _lambda):
    return GradientScaleLayer.apply(x, _lambda)
