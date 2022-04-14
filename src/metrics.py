def dice(input, target):    
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)

    return score.mean()


def iou(input, target):   
    axes = tuple(range(1, input.dim()))
    
    bin_input = (input > 0.5).int()
    target = target.int()
   
    intersect = (bin_input & target).float().sum(dim=axes)
    union = (bin_input | target).float().sum(dim=axes)
    score = intersect / (union + 1e-3)
   
    return score.mean()
    

def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()


if __name__ == '__main__':
    import torch
    metric = dice
    # metric = iou
    pred_y = torch.randn(size=(5, 2, 144, 144, 144), device='cpu') 
    true_y = torch.randn(size=(5, 2, 144, 144, 144), device='cpu') 

    print(metric(pred_y, true_y))