# python train.py --exp_name r1-lora-bert-sst2_kqv --task sst2
# python train.py --exp_name r1-lora-bert-cola_kqv --task cola
# python train.py --exp_name r1-lora-bert-mnli_kqv --task mnli

python train_stacked_loras.py --exp_name lora-bert-sst2_cola-trained-sst2 --task sst2