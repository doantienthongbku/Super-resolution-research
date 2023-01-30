import gc
import os
import queue
import threading

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import data.img_processing as imgproc

__all__ = [
    "TrainValidImageDataset", "TestImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]

class TrainValidImageDataset(Dataset):
    """
    Custom dataset for training and validation.
    
    Args:
        image_dir (str): Directory path of images.
        image_size (int): Size of images.
        mode (str): "train" or "valid".
    """
    def __init__(self, image_dir: str, image_size: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Get all image name from image_dir
        self.lr_image_file_names = [os.path.join(image_dir, "lr", file_name) for file_name in 
                                    os.listdir(os.path.join(image_dir, "lr"))]
        self.hr_image_file_names = [os.path.join(image_dir, "hr", file_name) for file_name in
                                    os.listdir(os.path.join(image_dir, "hr"))]
        # Specify the high-resolution image size, with equal length and width
        self.image_size = image_size
        self.mode = mode    # "train" or "valid"
        
        # Contains low-resolution and high-resolution image Tensor data
        self.lr_datasets = []
        self.hr_datasets = []
        
        # preload images into memory
        self.read_image_to_memory()

    def __getitem__(self, batch_index: int):
        # Read a batch of image data
        lr_y_image = self.lr_datasets[batch_index]
        hr_y_image = self.hr_datasets[batch_index]
        
        if self.mode == "train":
            # Data augment
            lr_y_image, hr_y_image = imgproc.random_crop(lr_y_image, hr_y_image, self.image_size)
            lr_y_image, hr_y_image = imgproc.random_rotate(lr_y_image, hr_y_image, angles=[0, 90, 180, 270])
            lr_y_image, hr_y_image = imgproc.random_horizontally_flip(lr_y_image, hr_y_image, p=0.5)
            lr_y_image, hr_y_image = imgproc.random_vertically_flip(lr_y_image, hr_y_image, p=0.5)
        elif self.mode == "valid":
            lr_y_image, hr_y_image = imgproc.center_crop(lr_y_image, hr_y_image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")
        
        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
        hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)

    def read_image_to_memory(self) -> None:
        lr_progress_bar = tqdm(self.lr_image_file_names,
                               total=len(self.lr_image_file_names),
                               unit="image",
                               desc=f"Read lr dataset into memory")
        
        # Disabling garbage collection after for loop helps speed things up
        gc.disable()
        
        for lr_image_file_name in lr_progress_bar:
            lr_image = cv2.imread(lr_image_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
            # Only extract the image data of the Y channel
            lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
            self.lr_datasets.append(lr_y_image)
            
        # Re-enable garbage collection
        gc.enable()
        
        hr_progress_bar = tqdm(self.hr_image_file_names,
                        total=len(self.hr_image_file_names),
                        unit="image",
                        desc=f"Read hr dataset into memory")

        for hr_image_file_name in hr_progress_bar:
            # Disabling garbage collection after for loop helps speed things up
            gc.disable()

            hr_image = cv2.imread(hr_image_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            # Only extract the image data of the Y channel
            hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)
            self.hr_datasets.append(hr_y_image)

            # After executing append, you need to turn on garbage collection again
            gc.enable()
            

class TestImageDataset(Dataset):
    def __init__(self, test_image_dir: str, upscale_factor: int) -> None:
        super(TestImageDataset, self).__init__()
        # Get all image file names in folder
        self.image_file_names = [os.path.join(test_image_dir, x) for x in os.listdir(test_image_dir)]
        # How many times the high-resolution image is the low-resolution image
        self.upscale_factor = upscale_factor

        # Contains low-resolution and high-resolution image Tensor data
        self.lr_datasets = []
        self.hr_datasets = []

        # preload images into memory
        self.read_image_to_memory()

    def __getitem__(self, batch_index: int):
        # Read a batch of image data
        lr_y_tensor = self.lr_datasets[batch_index]
        hr_y_tensor = self.hr_datasets[batch_index]

        return {"lr": lr_y_tensor, "hr": hr_y_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)

    def read_image_to_memory(self) -> None:
        progress_bar = tqdm(self.image_file_names,
                            total=len(self.image_file_names),
                            unit="image",
                            desc=f"Read test dataset into memory")

        for image_file_name in progress_bar:
            # Disabling garbage collection after for loop helps speed things up
            gc.disable()

            # Read a batch of image data
            hr_image = cv2.imread(image_file_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            # Use high-resolution image to make low-resolution image
            lr_image = imgproc.imresize(hr_image, 1 / self.upscale_factor)
            lr_image = imgproc.imresize(lr_image, self.upscale_factor)

            # Only extract the image data of the Y channel
            lr_y_image = imgproc.bgr2ycbcr(lr_image, use_y_channel=True)
            hr_y_image = imgproc.bgr2ycbcr(hr_image, use_y_channel=True)

            # Convert image data into Tensor stream format (PyTorch).
            # Note: The range of input and output is between [0, 1]
            lr_y_tensor = imgproc.image2tensor(lr_y_image, range_norm=False, half=False)
            hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=False)

            self.lr_datasets.append(lr_y_tensor)
            self.hr_datasets.append(hr_y_tensor)

            # After executing append, you need to turn on garbage collection again
            gc.enable()
            

class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.
    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.
    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)