import os
import sys
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

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import config.config as config
from data.data import CUDAPrefetcher
from data.data import TrainValidImageDataset, TestImageDataset
from model.model import VDSR
from support_func import *


def main():
    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load train dataset and valid dataset successfully.")
    
    model = build_model()
    print("Build VDSR model successfully.")
    
    psnr_criterion, pixel_criterion = define_loss()
    print("Define all loss functions successfully.")
    
    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")
    
    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")
    
    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Restore the parameters in the training node to this point
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict}
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")
        
    # Create a folder of super-resolution experiment results
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()
    
    for epoch in range(config.start_epoch, config.epochs):
        train(model, train_prefetcher, psnr_criterion, pixel_criterion, optimizer, epoch, scaler, writer)
        _ = validate(model, valid_prefetcher, psnr_criterion, epoch, writer, "Valid")
        psnr = validate(model, test_prefetcher, psnr_criterion, epoch, writer, "Test")
        print("\n")

        # Update lr
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"))
        
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"epoch_{epoch + 1}.pth.tar"), os.path.join(results_dir, "last.pth.tar"))
    
    
if __name__ == '__main__':
    main()
