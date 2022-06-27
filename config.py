"""docstring for CFG"""


class CFG:
    """_summary_"""

    wandb_key = "93460ff86561b201141546a407885ba3c8318d81"
    seed = 43
    data_dir = "./data/cifar10"
    pth_dir = "/pth"
    fold = 5
    n_split = 5
    num_workers = 2

    MODEL = "vgg"
    amp = True
    print_freq = 100
    batch_size = 128
    image_size = 128
    epochs = 30
    patience = 3

    accum_iter = 1
    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    max_grad_norm = 1000 if amp else 1
    max_lr = 1e-2
    min_lr = 1e-5
    weight_decay = 1e-6

    # augmix arg
    augmix = 1

    tta = 3
    weights = [1]
