# Weichao Qiu @ 2018
# Evaluation code to run hourglass
# Test the model trained from pytorch-pose
import torch
from mpii import Mpii
from hourglass import hg
import pdb
from evaluation import accuracy

def validate(val_loader, model, criterion, num_classes, debug = False, flip = False):
    # Check the performance
    # TODO: Why have the flip variable?

    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2) # 2 is for x and y

    model.eval()
    print('Set model to eval mode')

    for i, (inputs, target, meta) in enumerate(val_loader):
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(inputs.cuda(), volatile = True)
        target_var = torch.autograd.Variable(target, volatile = True)

        output = model(input_var)
        loss = 0
        for o in output:
            loss += criterion(o, target_var)

        score_map = output[-1].data.cpu()
        idx = [1,2,3,4,5,6,11,12,15,16]
        acc = accuracy(score_map, target.cpu(), idx)
        print(loss, acc)


def load_model():
    model_filename = 'mpii/hg_s2_b1/model_best.pth.tar'

    model = hg(num_stacks = 2, num_blocks = 1, num_classes = 16)
    model = torch.nn.DataParallel(model).cuda()

    sigma = 1
    label_type = 'Gaussian'
    test_batch_size = 6
    num_workers = 4 # Number of data loading workers
    num_classes = 16
    debug = True
    flip = False
    criterion = torch.nn.MSELoss(size_average=True).cuda()

    checkpoint = torch.load(model_filename)
    model.load_state_dict(checkpoint['state_dict'])

    print('Model weight loaded')

    val_loader = torch.utils.data.DataLoader(
        Mpii(
            'data/mpii/mpii_annotations.json', 'data/mpii/images',
            sigma = sigma, label_type = label_type, train = False),
        batch_size = test_batch_size, shuffle = False,
        num_workers = num_workers, pin_memory = True
    )
    print('Val loader created')

    loss, acc, predictions = validate(val_loader, model, criterion, num_classes, debug, flip)


def main():
    load_model()


if __name__ == '__main__':
    main()