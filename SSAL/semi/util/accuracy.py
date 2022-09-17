import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # print(output.shape)
        # print(target.shape)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(correct.shape)
        # print(correct[:1].shape)
        # print(correct[:1].float().sum(0, keepdim=True))
        # print(correct[:5].shape)
        # print(correct[:5].float().sum(0, keepdim=True))
        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0, keepdim=True).sum(-1, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # print(res)
        return res
