import os
import shutil
import time
from enum import Enum

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import config.config as config
from data.data import CUDAPrefetcher
from data.data import TrainValidImageDataset, TestImageDataset
from model.model import VDSR


# Copy form "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """compute and store the average and current value"""
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    

def load_dataset():
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, "train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, "valid")
    test_datasets = TestImageDataset(config.test_image_dir, config.upscale_factor)
    
    # Generate data loader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=False)
    
    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    
    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = VDSR().to(config.device)

    return model


def define_loss():
    pnsr_criterion = nn.MSELoss().to(config.device)
    pixel_criterion = nn.MSELoss(reduction="sum").to(config.device)
    
    return pnsr_criterion, pixel_criterion


def define_optimizer(model: nn.Module) -> optim.Optimizer:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay,
                          nesterov=config.model_nesterov)
    
    return optimizer


def define_scheduler(optimizer) -> lr_scheduler.StepLR:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_scheduler_step_size, gamma=config.lr_scheduler_gamma)

    return scheduler


def train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer):
    # Calculate how many iterations there are under epoch
    batches = len(train_prefetcher)
    
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    psnres = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generator in training mode
    model.train()
    
    batch_index = 0
    
    # Calculate the time it takes to test a batch of data
    end = time.time()
    
     # enable preload
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    
    while batch_data is not None:
        # mesure data loading time
        data_time.update(time.time() - end)
        
        lr = batch_data["lr"].to(config.device, non_blocking=True)
        hr = batch_data["hr"].to(config.device, non_blocking=True)
        
        # Initialize the generator gradient
        model.zero_grad()
        
        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Gradient zoom + gradient clipping
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.clip_gradient / optimizer.param_groups[0]["lr"], norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()
        
        # measure accuracy and record loss
        psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
        losses.update(loss.item(), lr.size(0))
        psnres.update(psnr.item(), lr.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Record training log information
        if batch_index % config.print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)
            
        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After a batch of data is calculated, add 1 to the number of batches
        batch_index += 1
        
    
def validate(model, valid_prefetcher, psnr_criterion, epoch, writer, mode) -> float:
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    psnres = AverageMeter("PSNR", ":4.2f", Summary.AVERAGE)
    progress = ProgressMeter(len(valid_prefetcher), [batch_time, psnres], prefix=f"{mode}: ")

    # Put the model in verification mode
    model.eval()

    batch_index = 0

    # Calculate the time it takes to test a batch of data
    end = time.time()
    with torch.no_grad():
        # enable preload
        valid_prefetcher.reset()
        batch_data = valid_prefetcher.next()

        while batch_data is not None:
            # measure data loading time
            lr = batch_data["lr"].to(config.device, non_blocking=True)
            hr = batch_data["hr"].to(config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(sr, hr))
            psnres.update(psnr.item(), lr.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % config.print_frequency == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = valid_prefetcher.next()

            # After a batch of data is calculated, add 1 to the number of batches
            batch_index += 1

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid":
        writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
    elif mode == "Test":
        writer.add_scalar("Test/PSNR", psnres.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg
