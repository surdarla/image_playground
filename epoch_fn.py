
from utils import *
from config import CFG
from tqdm.auto import tqdm
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(epoch,model, train_loader,criterion, optimizer,device,scaler,scheduler=None):
    model.train()
    losses = AverageMeter()
    start = end = time.time()

    for step, (images,targets) in tqdm(enumerate(train_loader),total=len(train_loader)):
      images = images.to(device)
      targets = targets.to(device)
      batch_size = targets.size(0)
      with autocast():
        out = model(images)
        loss = criterion(out, targets) 

      # if CFG.accum_iter > 1:
      #   loss = loss / CFG.accum_iter
      losses.update(loss.item(),batch_size)
      scaler.scale(loss).backward()

      if CFG.max_grad_norm: 
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),CFG.max_grad_norm) 
       
      # if ((step+1)%CFG.accum_iter==0): # or ((step+1)==len(train_loader)):
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()

      scheduler.step()

      end = time.time()
      if (step+1) % CFG.print_freq == 0 or step == (len(train_loader)-1):
        print('Epoch: [{0}][{1}/{2}] '
              'Elapsed {remain:s} '
              # 'Loss: {loss.val:.4f}({loss.avg:.4f}) '
              'Grad: {grad_norm:.4f}  '
              'LR: {lr:.6f}  '
              .format(epoch+1, step, len(train_loader), 
                      remain=timeSince(start, float(step+1)/len(train_loader)),
                      loss=losses,
                      grad_norm=grad_norm,
                      lr=scheduler.get_lr()[0]))
    
    return losses.avg


def valid_one_epoch(epoch,model, valid_loader, criterion,device,scheduler=None):
  model.eval()

  losses = AverageMeter()
  image_preds_all = []
  image_targets_all = []
  start = end = time.time()
  for step, (images, targets) in tqdm(enumerate(valid_loader),total=len(valid_loader)):
    images = images.to(device)
    targets = targets.to(device)
    batch_size = targets.size(0)
    y_preds = model(images)
    image_preds_all += [torch.argmax(y_preds,1).detach().to('cpu').numpy()]
    image_targets_all += [targets.detach().to('cpu').numpy()]
    loss = criterion(y_preds, targets)

    # if CFG.accum_iter > 1:
    #     loss = loss / CFG.accum_iter
    losses.update(loss.item(), batch_size)
    end = time.time()
    if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
        print('EVAL: [{0}/{1}] '
              'Elapsed {remain:s} '
              'Loss: {loss.val:.4f}({loss.avg:.4f}) '
              .format(step, len(valid_loader),
                      loss=losses,
                      remain=timeSince(start, float(step+1)/len(valid_loader))))
  image_preds_all = np.concatenate(image_preds_all)
  image_targets_all = np.concatenate(image_targets_all)
  val_score = (image_preds_all==image_targets_all).mean()
  return losses.avg, val_score

def inference_one_epoch(model, data_loader, device):
  model.eval()
  image_preds_all = []

  for step, (images, _) in tqdm(enumerate(data_loader),total=len(data_loader)):
      images = images.to(device).float()
      
      image_preds = model(images)  
      image_preds_all += [torch.softmax(image_preds, 1).detach().cpu().numpy()]
      
  image_preds_all = np.concatenate(image_preds_all, axis=0)
  return image_preds_all