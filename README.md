# PCLC

Code for our EMNLP2021 paper: [Bridge to Target Domain by Prototypical Contrastive Learning and Label Confusion: Re-explore Zero-Shot Learning for Slot Filling](https://arxiv.org/abs/2110.03572)

## Requirement

```
python==3.6.13
torch==1.4.0
cudatoolkit==10.1.243
cudnn==7.6.5
numpy==1.19.2
matplotlib==3.3.4
scikit-learn==0.24.2
scipy==1.5.4
tqdm==4.60.0
```

## Dataset

We use [SNIPS](https://github.com/MiuLab/SlotGated-SLU/tree/master/data/snips) dataset in our experiments, which has 7 domains and 39 slots types. We have divided the original dataset into seven sub-datasets according to their domains for the cross-domain slot filling task.  The whole dataset can be available at the `./data/snips`  folder.

## Run

### Configuration

- `--tgt_dm:` Target domain
- `--n_samples:` The number of samples in the target domain
- `--tr:` Template regularization flag
- `--emb_file:` Embedding file used in the experiment
- `--model_path:` Saved model path
- `--model_type:`Saved model type (e.g., pclc, ct, rzt)
- `--test_mode:`Choose mode to test the model (e.g., testset, seen_unseen)

### Train

Train `PCLC` in  `zero-shot` setting for the target domain `PlayMusic`:

```
python slu_main.py --exp_name path_to_model --exp_id pm_0   --bidirection --freeze_emb --tgt_dm PlayMusic  --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy
```

Train `PCLC` in  `few-shot` setting for the target domain `PlayMusic`:
```
python slu_main.py --exp_name path_to_model --exp_id pm_50   --bidirection --freeze_emb --tgt_dm PlayMusic  --n_samples 50 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy
```

Train  baseline model `CT` for the target domain `PlayMusic`:

```
python slu_baseline.py --exp_name ct --exp_id pm_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm PlayMusic --n_samples 0
```

### Test

Test `PCLC` on the target domain `PlayMusic` :

```
python slu_test.py --model_path ./experiments/path_to_model/pm_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm PlayMusic
```

Test `PCLC` on seen and unseen slots for the target domain `PlayMusic`

```
python slu_test.py --model_path ./experiments/path_to_model/pm_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm PlayMusic --test_mode seen_unseen
```

Test baseline model `CT` on the target domain `PlayMusic`:

```
python slu_test.py --model_path ./experiments/ct/pm_0/best_model.pth --model_type ct --n_samples 0 --tgt_dm PlayMusic
```

### Notes

- More details about the configurations can be found at [config.py](https://github.com/W-lw/PCLC/blob/main/config.py)
- A full set of commands can be found in [run.sh](https://github.com/W-lw/PCLC/blob/main/run.sh)
- All the models can be downloaded [here]()  to reproduce our results.

## Citation

 If you use any source codes or ideas included in this repository for your work, please cite the following paper.

```

```


