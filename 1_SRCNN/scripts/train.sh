python3 train.py --train-file "../datasets/91-image_x2.h5" \
                --eval-file "../datasets/Set5_x2.h5" \
                --outputs-dir "1_SRCNN/output" \
                --scale 2 \
                --lr 1e-4 \
                --batch-size 64 \
                --num-epochs 400 \
                --num-workers 8 \
                --seed 42 \
                --gpu-idx 0              