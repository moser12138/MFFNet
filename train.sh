## cityscapes
export CUDA_VISIBLE_DEVICES=0
cfg_file=configs/cityscapes.py
NGPUS=1
torchrun --nproc_per_node=$NGPUS tools/train_city.py --config $cfg_file

## camvid
python tools/train_camvid.py --cfg configs/camvid.yaml
