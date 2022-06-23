import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

from config import CFG
from utils import *
from data.prepare_data import (
    prepare_imgs_and_targets,
    MyDataset,
    transforms_train,
    transforms_valid,
)
from epoch_fn import inference_one_epoch
from model.myfish import Myfish


seed_everything(CFG.seed)
use_cuda = torch.cuda.is_available()
device = torch.cuda.device("cuda" if use_cuda else "cpu")

X_test, y_test = prepare_imgs_and_targets(CFG.data_dir, train=False)
testset = MyDataset(X_test, y_test, transform=transforms_valid(CFG.image_size))
ttaset = MyDataset(X_test, y_test, transform=transforms_train(CFG.image_size, params=1))
test_loader = DataLoader(
    testset,
    batch_size=CFG.batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=CFG.num_workers,
    drop_last=False,
)
tta_loader = DataLoader(
    ttaset,
    batch_size=CFG.batch_size,
    shuffle=False,
    pin_memory=False,
    num_workers=CFG.num_workers,
    drop_last=False,
)
paths = ["pth/final_0_best_8280.pth"]
models = []
for path in paths:
    #   checkpoint = torch.load(path)
    checkpoint = torch.load(path, map_location=torch.cuda.device("cpu"))
    model = Myfish()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    models.append(model)

ground_truth = [target for i, (_, target) in enumerate(testset)]
test_preds = []
tta_preds = []

for fold, model in enumerate(models):
    with torch.no_grad():
        for _ in range(CFG.tta):
            fold_weight_control = (CFG.weights[fold] / sum(CFG.weights)) / CFG.tta
            test_preds += [
                fold_weight_control * inference_one_epoch(model, test_loader, device)
            ]
            tta_preds += [
                fold_weight_control * inference_one_epoch(model, tta_loader, device)
            ]
    print(
        "fold {} normal loss = {:.5f}".format(fold, log_loss(ground_truth, test_preds))
    )
    print(
        "fold {} normal accuracy = {:.5f}".format(
            fold, (ground_truth == np.argmax(test_preds, axis=1)).mean()
        )
    )
    print("fold {} tta loss = {:.5f}".format(fold, log_loss(ground_truth, tta_preds)))
    print(
        "fold {} tta accuracy = {:.5f}".format(
            fold, (ground_truth == np.argmax(tta_preds, axis=1)).mean()
        )
    )

test_preds = np.mean(test_preds, axis=0)
tta_preds = np.mean(tta_preds, axis=0)
