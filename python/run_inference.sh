time=$(date '+%Y%m%d%H%M%S')

## input file
dataset_path="/home/ykhsieh/CV/final/dataset/"
checkpoint_path="/home/ykhsieh/CV/final/SEG2/log-20230529113854/checkpoints/model_best_982660.pth"

## output file
output_path="/home/ykhsieh/CV/final/output8"

bin="python3 inference.py "
CUDA_VISIBLE_DEVICES=0 $bin \
--dataset_path ${dataset_path} \
--checkpoint_path ${checkpoint_path} \
--output_path ${output_path} \