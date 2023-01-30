import os

# Prepare dataset
os.system("python3 ./prepare_dataset.py --images_dir ../../data_raw/BSDS200/original --output_dir ../datasets/BSDS200/VDSR/train --image_size 224 --step 42 --scale 2 --num_workers 10")
# os.system("python3 ./prepare_dataset.py --images_dir ../../data_raw/BSDS200/original --output_dir ../data/BSDS200/VDSR/train --image_size 42 --step 42 --scale 3 --num_workers 10")
# os.system("python3 ./prepare_dataset.py --images_dir ../../data_raw/BSDS200/original --output_dir ../data/BSDS200/VDSR/train --image_size 44 --step 44 --scale 4 --num_workers 10")

# Split train and valid
os.system("python3 ./split_train_valid_dataset.py --train_images_dir ../datasets/BSDS200/VDSR/train --valid_images_dir ../datasets/BSDS200/VDSR/valid --valid_samples_ratio 0.1")
