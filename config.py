import argparse
import json
def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-domain SLU")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="cross-domain-slu.log")
    parser.add_argument("--dump_path", type=str, default="experiments_test", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")

    # adaptation parameters
    parser.add_argument("--epoch", type=int, default=300, help="number of maximum epoch")
    parser.add_argument("--tgt_dm", type=str, default="", help="target_domain")
    parser.add_argument("--emb_file", type=str, default="./data/snips/emb/slu_word_char_embs.npy", help="embeddings file")  # slu_embs.npy w/o char embeddings   slu_word_char_embs.npy w/ char embeddings  slu_word_char_embs_with_slotembs.npy  w/ char and slot embs
    parser.add_argument("--emb_dim", type=int, default=400, help="embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--num_binslot", type=int, default=3, help="number of binary slot O,B,I")
    parser.add_argument("--num_slot", type=int, default=72, help="number of slot types")
    parser.add_argument("--num_domain", type=int, default=7, help="number of domain")
    parser.add_argument("--freeze_emb", default=False, action="store_true", help="Freeze embeddings")

    parser.add_argument("--slot_emb_file", type=str, default="./data/snips/emb/slot_word_char_embs_based_on_each_domain.dict", help="dictionary type: slot embeddings based on each domain") # slot_embs_based_on_each_domain.dict w/o char embeddings  slot_word_char_embs_based_on_each_domain.dict w/ char embeddings
    parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=200, help="hidden dimension for LSTM")
    parser.add_argument("--n_layer", type=int, default=2, help="number of layers for LSTM")
    parser.add_argument("--early_stop", type=int, default=5, help="No improvement after several epoch, we stop training")
    parser.add_argument("--binary", default=False, action="store_true", help="conduct binary training only")

    # add label_encoder
    parser.add_argument("--tr", default=False, action="store_true", help="use template regularization")

    # few shot learning
    parser.add_argument("--n_samples", type=int, default=0, help="number of samples for few shot learning")

    # encoder type for encoding entity tokens in the Step Two
    parser.add_argument("--enc_type", type=str, default="lstm", help="encoder type for encoding entity tokens (e.g., trs, lstm, none)")

    # transformer parameters
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads for transformer")
    parser.add_argument("--trs_hidden_dim", type=int, default=400, help="Dimension after combined into word level")
    parser.add_argument("--filter_size", type=int, default=64, help="Hidden size of the middle layer in FFN")
    parser.add_argument("--dim_key", type=int, default=0, help="Key dimension in transformer (if 0, then would be the same as hidden_size)")
    parser.add_argument("--dim_value", type=int, default=0, help="Value dimension in transformer (if 0, then would be the same as hidden_size)")
    parser.add_argument("--trs_layers", type=int, default=1, help="Number of layers for transformer")
########################################################################################################################
    #add contrastive loss for slot name predicting: 
    parser.add_argument("--use_contrastive_slotname_predictor", default=False, action="store_true")
    parser.add_argument("--use_final_predictor", default=False, action="store_true")
    # parser.add_argument("--add_contrastive_loss_for_slot", type=bool, default=True)
    # parser.add_argument("--add_contrastive_loss_for_context", default=True, action="store_false")
    # parser.add_argument("--add_mlp_for_slotname_context", default=True, action="store_false")
    # parser.add_argument("--add_mlp_for_slotname_slot", default=True, action="store_false")
    # parser.add_argument("--mlp_type", type=str, default="non_linear")
    
    # parser.add_argument("--add_context_feature", default=True, action="store_false")
    # parser.add_argument("--add_projection_for_slot", default=True, action="store_false")
    # parser.add_argument("--add_projection_for_context", default=True, action="store_false")
    # parser.add_argument("--contrastive_interdict", default=False, action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=0.1)


    # label smooth
    # parser.add_argument("--label_smooth", default=False, action="store_true")
    parser.add_argument("--smooth_factor", type=float, default=0.6)

    # contrastiveSlotNamePredictor parameters
    parser.add_argument("--slot_hidden_dim", type=int, default=400, help="number of domain")
    parser.add_argument("--context_hidden_dim", type=int, default=400, help="number of domain")
    parser.add_argument("--unseen_seen", type=str, default="", help="unseen/seen mode")

########################################################################################################################



    # baseline
    parser.add_argument("--use_example", default=False, action="store_true", help="use example value")
    parser.add_argument("--example_emb_file", type=str, default="./data/snips/emb/example_embs_based_on_each_domain.dict", help="dictionary type: example embeddings based on each domain")

    # test model
    parser.add_argument("--model_path", type=str, default="", help="Saved model path")
    parser.add_argument("--model_type", type=str, default="", help="Saved model type (e.g., coach, ct, rzt)")
    parser.add_argument("--test_mode", type=str, default="testset", help="Choose mode to test the model (e.g., testset, seen_unseen)")

    # NER
    parser.add_argument("--ner_entity_type_emb_file", type=str, default="./data/ner/emb/entity_type_embs.npy", help="entity type embeddings file path")
    parser.add_argument("--ner_example_emb_file", type=str, default="./data/ner/emb/example_embs.npy", help="entity example embeddings file path")
    parser.add_argument("--bilstmcrf", default=False, action="store_true", help="use BiLSTM-CRF baseline")
    parser.add_argument("--num_entity_label", type=int, default=9, help="number of entity label")

    params = parser.parse_args()
    
    
    return params
