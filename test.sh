CUDA_VISIBLE_DEVICES=0 python3 validate.py --arch=CLIP:ViT-L/14  --ckpt=pretrained_weights/fc_weights.pth --result_folder=clip_vitl14 



CUDA_VISIBLE_DEVICES=2 python3 inference_ojha.py


CUDA_VISIBLE_DEVICES=1 python3 inference_ojha.py