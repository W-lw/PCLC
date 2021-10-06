CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id scw_0   --bidirection --freeze_emb --tgt_dm SearchCreativeWork --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15 --theta 0.3

CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/scw_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm SearchCreativeWork >> scw.txt

CUDA_VISIBLE_DEVICES=0 python slu_main.py --exp_name final_test3 --exp_id sse_0   --bidirection --freeze_emb --tgt_dm SearchScreeningEvent --use_final_predictor --n_samples 0 --tr  --emb_file ./data/snips/emb/slu_word_char_embs_with_slotembs.npy   --early_stop 15 --alpha 0.8 --theta 0.5

CUDA_VISIBLE_DEVICES=0 python slu_test.py --model_path ./experiments_test/final_test3/sse_0/best_model.pth --model_type coach --n_samples 0 --tgt_dm SearchScreeningEvent	 >> sse.txt