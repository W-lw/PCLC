# CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id ap_0_alpha_0.4_smooth_0.9_use_early_15   --bidirection --freeze_emb --tgt_dm AddToPlaylist --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15    --smooth_factor 0.9 --alpha 0.4

# CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/ap_0_alpha_0.4_smooth_0.9_use_early_15/best_model.pth --model_type coach --n_samples 0 --tgt_dm AddToPlaylist   --smooth_factor 0.9 --alpha 0.4 >> ap.txt

# CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id br_0_alpha_0.1_smooth_0.6_use_early_15   --bidirection --freeze_emb --tgt_dm BookRestaurant --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15    --smooth_factor 0.6 --alpha 0.1

# CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/br_0_alpha_0.1_smooth_0.6_use_early_15/best_model.pth --model_type coach --n_samples 0 --tgt_dm BookRestaurant   --smooth_factor 0.6 --alpha 0.1 >> br.

# CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id gw_0   --bidirection --freeze_emb --tgt_dm GetWeather --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15    --smooth_factor 0.6 --alpha 0.1  --theta 0.5

# CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/gw_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm GetWeather >> gw.txt

CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id pm_0   --bidirection --freeze_emb --tgt_dm PlayMusic --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15 --theta 0.5

CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/pm_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm PlayMusic >> pm.txt

CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id rb_0   --bidirection --freeze_emb --tgt_dm RateBook --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15 --alpha 0.4

CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/rb_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm RateBook >> rb.txt