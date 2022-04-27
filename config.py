class CFG:
    amp=True
    seed=43
    # data_dir = '/content/drive/MyDrive/genesis_lab_cifar10/cifar-10-batches-py'
    data_dir = './data'
    wandb=True
    fold=5
    n_split=5
    num_workers = 2

    model = 'test'
    print_freq=100
    batch_size=256
    image_size=32
    epochs = 100
    patience=10

    accum_iter=1 # suppoprt to do batch accumulation for backprop with effectively larger batch size
    max_grad_norm=None
    max_lr = 1e-1
    min_lr = 1e-5
    weight_decay = 1e-6

    #augmix arg
    augmix=1

    tta=3
    weights=[1]