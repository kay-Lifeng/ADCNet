# ADCNet
# Environment
See OpenPCDet: https://github.com/open-mmlab/OpenPCDet

# Train exemple
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --launcher pytorch --cfg_file tools/cfgs/kitti_models/CaDDN.yaml

# Test exemple
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 tools/test.py --launcher pytorch --cfg_file tools/cfgs/kitti_models/CaDDN.yaml --batch_size 32 --ckpt output/cfgs/kitti_models/CaDDN/default/ckpt/checkpoint_epoch_80.pth
