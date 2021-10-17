#############################
####### Slot Filling ########
#############################

# train PCLC (both zero-shot and few-shot settings)
python slu_main.py --exp_name pclc --exp_id ap_0   --bidirection --freeze_emb --tgt_dm AddToPlaylist  --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id br_50   --bidirection --freeze_emb --tgt_dm BookRestaurant  --n_samples 50 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id gw_0   --bidirection --freeze_emb --tgt_dm GetWeather  --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id pm_50   --bidirection --freeze_emb --tgt_dm PlayMusic  --n_samples 50 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id rb_0   --bidirection --freeze_emb --tgt_dm RateBook  --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id scw_50   --bidirection --freeze_emb --tgt_dm SearchCreativeWork  --n_samples 50 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy

python slu_main.py --exp_name pclc --exp_id sse_0   --bidirection --freeze_emb --tgt_dm SearchScreeningEvent  --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy


# train baseline model (both on CT and RZT)
python slu_baseline.py --exp_name ct --exp_id ap_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm AddToPlaylist --n_samples 0 --model_type ct

python slu_baseline.py --exp_name rzt --exp_id br_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm BookRestaurant --n_samples 0 --model_type rzt

python slu_baseline.py --exp_name ct --exp_id gw_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm GetWeather --n_samples 50 --model_type ct

python slu_baseline.py --exp_name rzt --exp_id pm_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 200 --use_example --tgt_dm PlayMusic --n_samples 50 --model_type rzt

python slu_baseline.py --exp_name ct --exp_id rb_0 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm RateBook --n_samples 0 --model_type ct

python slu_baseline.py --exp_name rzt --exp_id sse_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --use_example --tgt_dm SearchScreeningEvent --n_samples 50 --model_type rzt

python slu_baseline.py --exp_name ct --exp_id scw_50 --bidirection --freeze_emb --lr 1e-4 --hidden_dim 300 --tgt_dm SearchCreativeWork --n_samples 50 --model_type ct

# test PCLC on testset
python slu_test.py --model_path ./experiments/pclc/ap_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm AddToPlaylist

python slu_test.py --model_path ./experiments/pclc/br_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm BookRestaurant

python slu_test.py --model_path ./experiments/pclc/gw_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm GetWeather

python slu_test.py --model_path ./experiments/pclc/pm_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm PlayMusic

python slu_test.py --model_path ./experiments/pclc/rb_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm RateBook

python slu_test.py --model_path ./experiments/pclc/scw_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm SearchCreativeWork

python slu_test.py --model_path ./experiments/pclc/sse_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm SearchScreeningEvent

# test PCLC on unseen and seen slots
python slu_test.py --model_path ./experiments/pclc/ap_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm AddToPlaylist --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/br_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm BookRestaurant --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/gw_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm GetWeather --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/pm_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm PlayMusic --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/rb_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm RateBook --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/scw_50/best_model.pth --model_type pclc --n_samples 50 --tgt_dm SearchCreativeWork --test_mode seen_unseen

python slu_test.py --model_path ./experiments/pclc/sse_0/best_model.pth --model_type pclc --n_samples 0 --tgt_dm SearchScreeningEvent --test_mode seen_unseen

# test baselines on testset
python slu_test.py --model_path ./experiments/ct/ap_0/best_model.pth --model_type ct --n_samples 0 --tgt_dm AddToPlaylist

python slu_test.py --model_path ./experiments/rzt/br_0/best_model.pth --model_type rzt --n_samples 0 --tgt_dm BookRestaurant

python slu_test.py --model_path ./experiments/ct/gw_50/best_model.pth --model_type ct --n_samples 50 --tgt_dm GetWeather

python slu_test.py --model_path ./experiments/rzt/pm_50/best_model.pth --model_type rzt --n_samples 50 --tgt_dm PlayMusic

python slu_test.py --model_path ./experiments/ct/rb_0/best_model.pth --model_type ct --n_samples 0 --tgt_dm RateBook

python slu_test.py --model_path ./experiments/rzt/scw_50/best_model.pth --model_type rzt --n_samples 50 --tgt_dm SearchCreativeWork

python slu_test.py --model_path ./experiments/ct/sse_50/best_model.pth --model_type ct --n_samples 50 --tgt_dm SearchScreeningEvent


# test baselines on unseen and seen slots
python slu_test.py --model_path ./experiments/ct/ap_0/best_model.pth --model_type ct --n_samples 0 --tgt_dm AddToPlaylist --test_mode seen_unseen

python slu_test.py --model_path ./experiments/rzt/br_0/best_model.pth --model_type rzt --n_samples 0 --tgt_dm BookRestaurant --test_mode seen_unseen

python slu_test.py --model_path ./experiments/ct/gw_50/best_model.pth --model_type ct --n_samples 50 --tgt_dm GetWeather --test_mode seen_unseen

python slu_test.py --model_path ./experiments/rzt/pm_50/best_model.pth --model_type rzt --n_samples 50 --tgt_dm PlayMusic --test_mode seen_unseen

python slu_test.py --model_path ./experiments/ct/rb_0/best_model.pth --model_type ct --n_samples 0 --tgt_dm RateBook --test_mode seen_unseen

python slu_test.py --model_path ./experiments/rzt/scw_50/best_model.pth --model_type rzt --n_samples 50 --tgt_dm SearchCreativeWork --test_mode seen_unseen

python slu_test.py --model_path ./experiments/ct/sse_50/best_model.pth --model_type ct --n_samples 50 --tgt_dm SearchScreeningEvent --test_mode seen_unseen
