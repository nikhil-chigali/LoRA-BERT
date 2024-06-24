# LoRA-Ensembling

### Trained Bert model with rank-1 LoRA weights

Trained Bert model with rank-1 LoRA weights on `key`, `query`, and `value` weights. Training hyperparameter settings can be found under `hparams/` folder. Training progress can be found on the [wandb report](https://wandb.ai/nikhilchigali/LoRA-Ensembling/reports/Training-LoRA-models--Vmlldzo4NDI2MjM2)

#### Tasks trained,
1. [GLUE-SST2](https://huggingface.co/datasets/nyu-mll/glue#sst2-1)
```
sentence: a string feature.
label: a classification label, with possible values including negative (0), positive (1).
idx: a int32 feature.
```

2. [GLUE-CoLA](https://huggingface.co/datasets/nyu-mll/glue#cola-1)
```
sentence: a string feature.
label: a classification label, with possible values including unacceptable (0), acceptable (1).
idx: a int32 feature.
```
3. [GLUE-MNLI](https://huggingface.co/datasets/nyu-mll/glue#mnli-1)
```
premise: a string feature.
hypothesis: a string feature.
label: a classification label, with possible values including entailment (0), neutral (1), contradiction (2).
idx: a int32 feature.
```

**Baseline metrics:**

```python
sst2_baseline = {'train/loss': 0.3071436285972595, 'train/acc': 0.8725202083587646,
                 'val/loss': 0.3317883610725403, 'val/acc': 0.8463302850723267}

cola_baseline = {'train/loss': 0.23546119034290314, 'train/acc': 0.9183052778244019,
                 'val/loss': 0.41714978218078613, 'val/acc': 0.8144230842590332}
mnli_baseline = [{'val_matched/loss/dataloader_idx_0': 0.6197290420532227,
  'val_matched/acc/dataloader_idx_0': 0.7426388263702393},
 {'val_mismatched/loss/dataloader_idx_1': 0.5963663458824158,
  'val_mismatched/acc/dataloader_idx_1': 0.7571195960044861}]

```

Refer to *observations.ipynb* for more details

- Compare performance difference between stacked LoRA's on the original tasks.

## Work in Progress

- Comparing change in LoRA weights using metrics like Cosine Similarity
- Measure subspace similarity between LoRA's of different tasks
![alt text](images/subspace-sim.png)
- Try out various methods to train the stacked rank-1 LoRA's for better generalizability on both the tasks.
- Try to implement filter/gates for training Mixture of LoRA Experts.