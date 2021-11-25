from src.modules import Lstm, CRF, Attention
from src.slu.datareader import SLOT_PAD, y2_set, domain_set
# from src.transformer import TransformerEncoder
from src.utils import load_embedding_from_npy, load_embedding_from_pkl
from preprocess.gen_embeddings_for_slu import domain2slot
import torch
from torch import cosine_similarity, nn
from torch.nn import functional as F
import pdb
import numpy as np
class BinarySLUTagger(nn.Module):
    def __init__(self, params, vocab):
        super(BinarySLUTagger, self).__init__()
        
        self.lstm = Lstm(params, vocab)
        self.num_binslot = params.num_binslot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_binslot)
        self.crf_layer = CRF(self.num_binslot)
        
    def forward(self, X, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_binslot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction, lstm_hidden
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y


class SlotNamePredictor(nn.Module):
    def __init__(self, params):
        super(SlotNamePredictor, self).__init__()
        self.params = params
        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.enc_type = params.enc_type
        self.lstm_enc = nn.LSTM(self.input_dim, params.trs_hidden_dim//2, num_layers=params.trs_layers, bidirectional=True, batch_first=True)
        
        self.crossLoss = nn.CrossEntropyLoss(reduction='sum')
        self.smoothLoss = nn.KLDivLoss(reduction='sum')

        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)
        self.all_slots_embedding = {}
        for key, values in self.slot_embs.items():
            for idx, slot_embedding in enumerate(values):
                self.all_slots_embedding[domain2slot[key][idx]] = slot_embedding
        self.similarity_matrix = self.getSlotSimilarity()

    def getSlotSimilarity(self, sim_type="cosine"):
        similarity_matrix = {}
        all_slots = list(self.all_slots_embedding.keys())
        def cos_similarity(x,y):
            num = x.dot(y.T)
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            return (num / denom)
        
        def euclid_similarity(x,y):
            return round(np.linalg.norm(x-y),2)
        
        for i in range(len(all_slots)):
            cur_slot = all_slots[i]
            similarity_matrix[cur_slot] = {}
            for j in range(len(all_slots)):
                cor_slot = all_slots[j]
                if sim_type == "cosine":
                    similarity = cos_similarity(self.all_slots_embedding[cur_slot], self.all_slots_embedding[cor_slot])
                elif sim_type == "euclid":
                    similarity = euclid_similarity(self.all_slots_embedding[cur_slot], self.all_slots_embedding[cor_slot])
                similarity_matrix[cur_slot][cor_slot] = similarity

        return similarity_matrix

    def getSmoothLabel(self, slot_name, slot_list_based_domain):
        target_domain = self.params.tgt_dm
        slot_list_target = domain2slot[target_domain]
        num_classes = len(slot_list_based_domain) + len(slot_list_target)
        label = torch.LongTensor([slot_list_based_domain.index(slot_name)])
        one_hot_label = nn.functional.one_hot(label, num_classes=len(slot_list_based_domain))
        smooth_factor = self.params.smooth_factor
        one_hot_label = (one_hot_label*smooth_factor).squeeze()
        slots_similarity = []
        for slot in slot_list_target:
            similarity = self.similarity_matrix[slot_name][slot]
            slots_similarity.append(similarity)
        slots_similarity = torch.FloatTensor(slots_similarity)
        # slots_similarity =  nn.functional.log_softmax(slots_similarity)
        slots_similarity = slots_similarity*(1.0-smooth_factor)
        smooth_label = torch.cat((one_hot_label, slots_similarity))
        return smooth_label
        
    def forward(self, domains, hidden_layers, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []

        bsz = domains.size()[0]
        
        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
            
            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            # get indices of B and I
            B_list = bin_label == 1
            I_list = bin_label == 2
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        # indices = prev_index
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)

                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    # slot_feats.squeeze(0) ==> (hidden_dim)
                    feature_each_sample.append(slot_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]
                        gold_smooth_label = self.getSmoothLabel(slot_name, slot_list_based_domain)
                        assert gold_smooth_label.dtype == torch.float32
                        gold_slotname_each_sample.append(gold_smooth_label)
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    feature_each_sample.append(slot_feats.squeeze(0))

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[curr_index]].split("-")[1]
                        gold_smooth_label = self.getSmoothLabel(slot_name, slot_list_based_domain)
                        assert gold_smooth_label.dtype == torch.float32
                        gold_slotname_each_sample.append(gold_smooth_label)
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                        
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            feature_list.append(feature_each_sample)
            if final_golds is not None:
                gold_slotname_each_sample = torch.stack(gold_slotname_each_sample)
                gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)

        ### predict slot names
        
        pred_slotname_list = []
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            if final_golds is not None:
                # w/ label smooth
                slot_embs_source_domain = torch.FloatTensor(self.slot_embs[domain_name]).transpose(0,1).cuda()
                slot_embs_target_domain = torch.FloatTensor(self.slot_embs[self.params.tgt_dm]).transpose(0,1).cuda()
                slot_embs_based_domain = torch.cat((slot_embs_source_domain, slot_embs_target_domain), dim=1)
            else:
                # w/o label smooth
                slot_embs_based_domain = torch.FloatTensor(self.slot_embs[domain_name]).transpose(0,1).cuda()   # (emb_dim, slot_num)

            feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                pred_slotname_each_sample = torch.matmul(feature_each_sample, slot_embs_based_domain) # (num_slotname, slot_num)
                
            
            pred_slotname_list.append(pred_slotname_each_sample)

        ### caculate the loss for slot name
        if final_golds is not None:
            loss_slotname = 0.0
            slot_num = 0.0
            for pred_slotname_each_sample, gold_slotname_each_sample in zip(pred_slotname_list, gold_slotname_list):

                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                gold_slotname_each_sample = gold_slotname_each_sample.cuda()
                slot_num += gold_slotname_each_sample.size()[0]
                pred_slotname_each_sample = nn.functional.log_softmax(pred_slotname_each_sample)
                loss_slotname += self.smoothLoss(pred_slotname_each_sample, gold_slotname_each_sample)
                loss_slotname += self.crossLoss(pred_slotname_each_sample, gold_slotname_each_sample)
            
            loss_slotname = loss_slotname/slot_num

        if final_golds is not None:
            # only in the training mode
            return loss_slotname
        else:
            return pred_slotname_list




class ContrastiveSlotNamePredictor(nn.Module):
    def __init__(self, params) -> None:
        super(ContrastiveSlotNamePredictor, self).__init__()
        self.params = params
        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.enc_type = params.enc_type
        self.lstm_enc = nn.LSTM(self.input_dim, params.trs_hidden_dim//2, num_layers=params.trs_layers, bidirectional=True, batch_first=True)
        
        self.crossLoss = nn.CrossEntropyLoss(reduction='sum')

        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)#slot_num * 400； keys 为各个领域名字
        self.slot_name_projection_for_context = nn.Linear(params.emb_dim, params.context_hidden_dim, bias=False)
        self.slot_name_projection_for_slot = nn.Linear(params.emb_dim, params.slot_hidden_dim, bias=False)
        self.slot_projection = nn.Linear(params.trs_hidden_dim, params.slot_hidden_dim)
        self.all_slots_embedding = {}
        for key, values in self.slot_embs.items():
            for idx, slot_embedding in enumerate(values):
                self.all_slots_embedding[domain2slot[key][idx]] = slot_embedding
        self.similarity_matrix = self.getSlotSimilarity()

    def getSlotSimilarity(self, sim_type="cosine"):
        similarity_matrix = {}
        all_slots = list(self.slot_embs.keys())
        def cos_similarity(x,y):
            num = x.dot(y.T)
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            return (num / denom)
        
        def euclid_similarity(x,y):
            return round(np.linalg.norm(x-y),2)
        
        for i in range(len(all_slots)):
            cur_slot = all_slots[i]
            similarity_matrix[cur_slot] = {}
            for j in range(len(all_slots)):
                cor_slot = all_slots[j]
                if sim_type == "cosine":
                    similarity = cos_similarity(self.slot_embs[cur_slot], self.slot_embs[cor_slot])
                elif sim_type == "euclid":
                    similarity = euclid_similarity(self.slot_embs[cur_slot], self.slot_embs[cor_slot])
                similarity_matrix[cur_slot][cor_slot] = similarity
        return similarity_matrix

    def getAllSlotEmbedding(self, target_slot, domain2slot):
        keys = [torch.FloatTensor(self.all_slots_embedding[target_slot])]
        neg_embedding = [torch.FloatTensor(value) for key,value in self.all_slots_embedding.items() if key != target_slot]
        keys.extend(neg_embedding)
        allSlotEmbedding = torch.stack(keys).cuda()
        return allSlotEmbedding

    def forward(self, domains, hidden_layers, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []
            slotname_unencode = []
        bsz = domains.size()[0]
        
        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names

            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
                slotname_unencode_each_sample = []
            
            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            # get indices of B and I
            B_list = bin_label == 1
            I_list = bin_label == 2
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    # slot_feats.squeeze(0) ==> (hidden_dim)
                    feature_each_sample.append(slot_feats.squeeze(0))
                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                        slotname_unencode_each_sample.append(slot_name)

                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    feature_each_sample.append(slot_feats.squeeze(0))

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[curr_index]].split("-")[1]
                        gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                        slotname_unencode_each_sample.append(slot_name)
                        
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            feature_list.append(feature_each_sample)
            if final_golds is not None:
                gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)
                slotname_unencode.append(slotname_unencode_each_sample)

        ### predict slot names
        pred_slotname_list = []

        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            
            feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                slot_embs_based_domain = torch.FloatTensor(self.slot_embs[domain_name]).cuda()   # (emb_dim, slot_num)
                slot_embs_based_domain = self.slot_name_projection_for_slot(slot_embs_based_domain).transpose(0,1)
                feature_each_sample = self.slot_projection(feature_each_sample)
                pred_slotname_each_sample = torch.matmul(feature_each_sample, slot_embs_based_domain) # (num_slotname, slot_num)
            
            pred_slotname_list.append(pred_slotname_each_sample)

        ### caculate the loss for slot name
        if final_golds is not None:
            loss_slotname = 0.0
            slot_num = 0.0
            for pred_slotname_each_sample, gold_slotname_each_sample in zip(pred_slotname_list, gold_slotname_list):

                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                gold_slotname_each_sample = gold_slotname_each_sample.cuda()
                slot_num += gold_slotname_each_sample.size()[0]
                pred_slotname_each_sample = nn.functional.log_softmax(pred_slotname_each_sample)
                loss_slotname += self.smoothLoss(pred_slotname_each_sample, gold_slotname_each_sample)
                loss_slotname += self.crossLoss(pred_slotname_each_sample, gold_slotname_each_sample)
            
            loss_slotname = loss_slotname/slot_num

        ### get contrastive loss for slot embedding
        if final_golds is not None:
            slot_contrastive_loss = 0.0
            slot_num = 0.0
            for i in range(bsz):
                feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
                slotname_unencode_each_sample = slotname_unencode[i]
                assert len(feature_each_sample) == len(slotname_unencode_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                else:
                    for j in range(len(feature_each_sample)):
                        slot_name = slotname_unencode_each_sample[j]
                        all_slot_embeddings = self.getAllSlotEmbedding(slot_name, domain2slot)
                        all_slot_proj = self.slot_name_projection_for_slot(all_slot_embeddings)
                        feature = self.slot_projection(feature_each_sample[j])
                        cont_loss = contrastive_loss(feature, all_slot_proj, self.params.temperature)
                        slot_contrastive_loss += cont_loss
                        slot_num += 1.0
            slot_contrastive_loss = slot_contrastive_loss/slot_num


        if final_golds is not None:
            # only in the training mode
            return loss_slotname, slot_contrastive_loss
        else:
            return pred_slotname_list


class FinalSlotNamePredictor(nn.Module):
    def __init__(self, params) -> None:
        super(FinalSlotNamePredictor, self).__init__()
        self.params = params
        self.input_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.enc_type = params.enc_type
        self.lstm_enc = nn.LSTM(self.input_dim, params.trs_hidden_dim//2, num_layers=params.trs_layers, bidirectional=True, batch_first=True)
        
        self.crossLoss = nn.CrossEntropyLoss(reduction='sum')
        self.smoothLoss = nn.KLDivLoss(reduction='sum')
        self.slot_embs = load_embedding_from_pkl(params.slot_emb_file)#slot_num * 400； keys 为各个领域名字

        linear1 = nn.Linear(params.emb_dim, params.emb_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(params.emb_dim, params.context_hidden_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)

        self.slot_name_projection_for_context = nn.Sequential(linear1, activation, drop, linear2)
        

        linear1 = nn.Linear(params.emb_dim, params.emb_dim, bias=True)
        activation = nn.Tanh()
        drop = nn.Dropout(0.2)
        linear2 = nn.Linear(params.emb_dim, params.context_hidden_dim, bias=True)
        
        nn.init.eye_(linear1.weight)
        nn.init.constant_(linear1.bias, 0.0)
        nn.init.eye_(linear2.weight)
        nn.init.constant_(linear2.bias, 0.0)
        self.slot_name_projection_for_slot = nn.Sequential(linear1, activation, drop, linear2)

        linear1 = nn.Linear(params.emb_dim, params.emb_dim, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(params.emb_dim, params.context_hidden_dim, bias=True)
        self.slot_projection = nn.Sequential(linear1, activation, linear2)

        linear1 = nn.Linear(params.emb_dim, params.emb_dim, bias=True)
        activation = nn.Tanh()
        linear2 = nn.Linear(params.emb_dim, params.context_hidden_dim, bias=True)
        self.context_projection = nn.Sequential(linear1, activation, linear2)
        
        self.all_slots_embedding = {}
        for key, values in self.slot_embs.items():
            for idx, slot_embedding in enumerate(values):
                self.all_slots_embedding[domain2slot[key][idx]] = torch.FloatTensor(slot_embedding)

    def getSmoothLabel(self, slot_name, source_domain_name):
        target_domain_name = self.params.tgt_dm
        
        slot_embedding_unencode = self.all_slots_embedding[slot_name].unsqueeze(dim=0).cuda()#(1,emb_dim)
        slot_embs_based_domain_target = torch.FloatTensor(self.slot_embs[target_domain_name]).cuda()    # (slot_numm, emb_dim)

        slot_embedding_context = self.slot_name_projection_for_context(slot_embedding_unencode)
        slot_embs_based_domain_target_context = self.slot_name_projection_for_context(slot_embs_based_domain_target)

        slot_embedding_slot = self.slot_name_projection_for_slot(slot_embedding_unencode)
        slot_embs_based_domain_target_slot = self.slot_name_projection_for_slot(slot_embs_based_domain_target)
        slot_embedding = torch.cat((slot_embedding_slot,slot_embedding_context), dim=1)#(1,emb_dim*2)
        slot_embs_based_domain = torch.cat((slot_embs_based_domain_target_slot, slot_embs_based_domain_target_context), dim=1)#(slot_num,emb_dim*2)

        slot_embedding = slot_embedding.expand(slot_embs_based_domain.size())
        slot_embedding = slot_embedding.detach()
        slot_embs_based_domain = slot_embs_based_domain.detach()
        slots_similarity = torch.cosine_similarity(slot_embedding, slot_embs_based_domain, dim=1)
        assert len(slots_similarity) == len(slot_embs_based_domain)

        # target_domain = self.params.tgt_dm
        # slot_list_target = domain2slot[target_domain]
        slot_list_based_domain = domain2slot[source_domain_name]
        label = torch.LongTensor([slot_list_based_domain.index(slot_name)])
        one_hot_label = nn.functional.one_hot(label, num_classes=len(slot_list_based_domain))

        smooth_factor = self.params.smooth_factor
        one_hot_label = (one_hot_label*smooth_factor).squeeze().cuda()
        slots_similarity = slots_similarity / slots_similarity.sum()
        slots_similarity = slots_similarity*(1.0-smooth_factor)
        smooth_label = torch.cat((one_hot_label, slots_similarity))
        return smooth_label
    
    def getAllSlotEmbedding(self, target_slot, domain2slot):
        keys = [torch.FloatTensor(self.all_slots_embedding[target_slot])]
        neg_embedding = [torch.FloatTensor(value) for key,value in self.all_slots_embedding.items() if key != target_slot]
        keys.extend(neg_embedding)
        allSlotEmbedding = torch.stack(keys).cuda()
        return allSlotEmbedding

    def getContextEmbedding(self, current_slot_feats, hiddens, valid_length, indices):
        """
            get the Context embedding by attention

            current_slot_feats: (1, hidden_dim);
            hiddens: (seq_len, hidden_dim);
            valid_length: length of valid part of the sequence;
            indices: indices of slot in the sequence; (slot_len)
        """
        current_slot_feats = current_slot_feats.detach()
        index = torch.arange(hiddens.shape[0])
        mask = (index<indices[0])| ((index>indices[-1]) & (index< valid_length))
        mask = mask.cuda()
        masked_weights = hiddens.matmul(current_slot_feats.squeeze()) * mask
        unnorm_weights = (masked_weights - masked_weights.max()).exp()
        norm_weights = unnorm_weights / unnorm_weights.sum()
        
        attention_weights = torch.mul(hiddens, norm_weights.unsqueeze(-1))
        context_embedding = attention_weights.sum(dim=0)

        return context_embedding
    
    def forward(self, domains, hidden_layers, lengths=None, epoch=None, binary_preditions=None, binary_golds=None, final_golds=None):
        """
        Inputs:
            domains: domain list for each sample (bsz,)
            hidden_layers: hidden layers from encoder (bsz, seq_len, hidden_dim)
            lengths: length of every example (bsz,)
            binary_predictions: predictions made by our model (bsz, seq_len)
            binary_golds: in the teacher forcing mode: binary_golds is not None (bsz, seq_len)
            final_golds: used only in the training mode (bsz, seq_len)
        Outputs:
            pred_slotname_list: list of predicted slot names
            gold_slotname_list: list of gold slot names  (only return this in the training mode)
        """
        binary_labels = binary_golds if binary_golds is not None else binary_preditions

        feature_list = []
        context_feature_list = []
        if final_golds is not None:
            # only in the training mode
            gold_slotname_list = []
            slotname_unencode = []
        bsz = domains.size()[0]

        ### collect features of slot and their corresponding labels (gold_slotname) in this batch
        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            slot_list_based_domain = domain2slot[domain_name]  # a list of slot names
            valid_length = lengths[i]
            # we can also add domain embeddings after transformer encoder
            hidden_i = hidden_layers[i]    # (seq_len, hidden_dim)

            ## collect range of slot name and hidden layers
            feature_each_sample = []
            context_feature_each_sample = []
            if final_golds is not None:
                final_gold_each_sample = final_golds[i]
                gold_slotname_each_sample = []
                slotname_unencode_each_sample = []
            
            bin_label = binary_labels[i]
            bin_label = torch.LongTensor(bin_label)
            # get indices of B and I
            B_list = bin_label == 1
            I_list = bin_label == 2
            nonzero_B = torch.nonzero(B_list)
            num_slotname = nonzero_B.size()[0]
            
            if num_slotname == 0:
                feature_list.append(feature_each_sample)
                context_feature_list.append(context_feature_each_sample)
                continue

            for j in range(num_slotname):
                if j == 0 and j < num_slotname-1:
                    prev_index = nonzero_B[j]
                    continue

                curr_index = nonzero_B[j]
                if not (j == 0 and j == num_slotname-1):
                    nonzero_I = torch.nonzero(I_list[prev_index: curr_index])
                    indices = None
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + prev_index).squeeze(1) # squeeze to one dimension
                        indices = torch.cat((prev_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        indices = prev_index
                        hiddens_based_slotname = hidden_i[prev_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1) # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    # slot_feats.squeeze(0) ==> (hidden_dim)
                    context_feats = self.getContextEmbedding(slot_feats, hidden_i, valid_length, indices)
                    context_feature_each_sample.append(context_feats)
                    feature_each_sample.append(slot_feats.squeeze(0))

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[prev_index]].split("-")[1]
                        slotname_unencode_each_sample.append(slot_name)

                        if self.params.tgt_dm != domain_name:
                            gold_smooth_label = self.getSmoothLabel(slot_name, domain_name)
                            gold_slotname_each_sample.append(gold_smooth_label)
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))

                
                if j == num_slotname - 1:
                    nonzero_I = torch.nonzero(I_list[curr_index:])
                    indices = None
                    if len(nonzero_I) != 0:
                        nonzero_I = (nonzero_I + curr_index).squeeze(1)  # squeeze to one dimension
                        indices = torch.cat((curr_index, nonzero_I), dim=0)
                        hiddens_based_slotname = hidden_i[indices.unsqueeze(0)]   # (1, subseq_len, hidden_dim)
                    else:
                        # length of slot name is only 1
                        indices = curr_index
                        hiddens_based_slotname = hidden_i[curr_index.unsqueeze(0)]  # (1, 1, hidden_dim)
                    
                    if self.enc_type == "lstm":
                        slot_feats, (_, _) = self.lstm_enc(hiddens_based_slotname)   # (1, subseq_len, hidden_dim)
                        slot_feats = torch.sum(slot_feats, dim=1)  # (1, hidden_dim)
                    else:
                        slot_feats = torch.sum(hiddens_based_slotname, dim=1) # (1, hidden_dim)
                    # slot_feats = torch.sum(slot_feats, dim=1)
                    # if self.params.add_context_feature: 
                    context_feats = self.getContextEmbedding(slot_feats, hidden_i, valid_length, indices)
                    context_feature_each_sample.append(context_feats)

                    feature_each_sample.append(slot_feats.squeeze(0))

                    if final_golds is not None:
                        slot_name = y2_set[final_gold_each_sample[curr_index]].split("-")[1]
                        slotname_unencode_each_sample.append(slot_name)
                        if self.params.tgt_dm != domain_name:
                            gold_smooth_label = self.getSmoothLabel(slot_name, domain_name)
                            gold_slotname_each_sample.append(gold_smooth_label)
                        else:
                            gold_slotname_each_sample.append(slot_list_based_domain.index(slot_name))
                        
                else:
                    prev_index = curr_index
            
            feature_each_sample = torch.stack(feature_each_sample)  # (num_slotname, hidden_dim)
            context_feature_each_sample = torch.stack(context_feature_each_sample)
            feature_list.append(feature_each_sample)
            context_feature_list.append(context_feature_each_sample)
            if final_golds is not None:
                if self.params.tgt_dm != domain_name:
                    gold_slotname_each_sample = torch.stack(gold_slotname_each_sample)
                else:
                    gold_slotname_each_sample = torch.LongTensor(gold_slotname_each_sample)   # (num_slotname)
                gold_slotname_list.append(gold_slotname_each_sample)
                slotname_unencode.append(slotname_unencode_each_sample)
        ### predict slot names
        '''
            to add label smooth
        '''
        pred_slotname_list = []

        for i in range(bsz):
            dm_id = domains[i]
            domain_name = domain_set[dm_id]
            
            feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
            
            context_feature_each_sample = context_feature_list[i]
            assert len(feature_each_sample) == len(context_feature_each_sample)
            if len(feature_each_sample) == 0:
                # only in the evaluation phrase
                pred_slotname_each_sample = None
            else:
                
                if self.params.tgt_dm != domain_name and final_golds is not None:
                    # w/ label smooth
                    slot_embs_source_domain = torch.FloatTensor(self.slot_embs[domain_name])
                    slot_embs_target_domain = torch.FloatTensor(self.slot_embs[self.params.tgt_dm])
                    slot_embs_based_domain_source = torch.cat((slot_embs_source_domain, slot_embs_target_domain), dim=0).cuda()
                else:
                    slot_embs_based_domain_source = torch.FloatTensor(self.slot_embs[domain_name]).cuda()   # ( slot_num,emb_dim)

                slot_embs_based_domain = self.slot_name_projection_for_slot(slot_embs_based_domain_source).transpose(0,1)

                feature_each_sample = self.slot_projection(feature_each_sample)


                slot_embs_based_domain_context = self.slot_name_projection_for_context(slot_embs_based_domain_source).transpose(0,1)
                context_feature_each_sample = self.context_projection(context_feature_each_sample)
                pred_slotname_each_sample = torch.matmul(feature_each_sample, slot_embs_based_domain) # (num_slotname, slot_num)
                pred_slotname_each_sample_context = torch.matmul(context_feature_each_sample, slot_embs_based_domain_context)
                pred_slotname_each_sample = pred_slotname_each_sample + self.params.theta*pred_slotname_each_sample_context 
            pred_slotname_list.append(pred_slotname_each_sample)

        ### caculate the loss for slot name
        if final_golds is not None:
            loss_slotname = 0.0
            slot_num = 0.0
            
            for pred_slotname_each_sample, gold_slotname_each_sample, domain_id in zip(pred_slotname_list, gold_slotname_list, domains):
                domain_name = domain_set[domain_id]
                assert pred_slotname_each_sample.size()[0] == gold_slotname_each_sample.size()[0]
                gold_slotname_each_sample = gold_slotname_each_sample.cuda()
                # loss_slotname = self.loss_fn(pred_slotname_each_sample, gold_slotname_each_sample.cuda())
                slot_num += gold_slotname_each_sample.size()[0]
                if self.params.tgt_dm != domain_name:
                    pred_slotname_each_sample = nn.functional.log_softmax(pred_slotname_each_sample)
                    loss_slotname += self.smoothLoss(pred_slotname_each_sample, gold_slotname_each_sample)

                else:
                    loss_slotname += self.crossLoss(pred_slotname_each_sample, gold_slotname_each_sample)
            
            loss_slotname = loss_slotname/slot_num

        ### four kinds of contrastive loss
        if final_golds is not None:
            # calculate the contrastive loss for slot embedding with slot name embedding
            slot_contrastive_loss = 0.0
            slot_num = 0.0
            for i in range(bsz):
                feature_each_sample = feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
                slotname_unencode_each_sample = slotname_unencode[i]
                assert len(feature_each_sample) == len(slotname_unencode_each_sample)
                if len(feature_each_sample) == 0:
                    continue
                else:
                    for j in range(len(feature_each_sample)):
                        slot_name = slotname_unencode_each_sample[j]
                        all_slot_embeddings = self.getAllSlotEmbedding(slot_name, domain2slot)
                        all_slot_proj = self.slot_name_projection_for_slot(all_slot_embeddings)
                        feature = self.slot_projection(feature_each_sample[j])

                        feature = feature.detach()
                        cont_loss = contrastive_loss(feature, all_slot_proj, self.params.temperature)
                        slot_contrastive_loss += cont_loss
                        slot_num += 1.0
            slot_contrastive_loss = slot_contrastive_loss/slot_num
            ### get contrastive loss for context embedding
            context_contrastive_loss = None
            context_contrastive_loss = 0.0
            context_num = 0.0
            for i in range(bsz):
                context_feature_each_sample = context_feature_list[i]  # (num_slotname, hidden_dim)  hidden_dim == emb_dim
                slotname_unencode_each_sample = slotname_unencode[i]
                assert len(context_feature_each_sample) == len(slotname_unencode_each_sample)
                if len(context_feature_each_sample) == 0:
                    continue
                else:
                    for j in range(len(context_feature_each_sample)):
                        slot_name = slotname_unencode_each_sample[j]
                        all_slot_embeddings = self.getAllSlotEmbedding(slot_name, domain2slot)
                        all_slot_proj = self.slot_name_projection_for_context(all_slot_embeddings)
                        context_feature = self.context_projection(context_feature_each_sample[j])
                        context_feature = context_feature.detach()
                        cont_loss = contrastive_loss(context_feature, all_slot_proj, self.params.temperature)
                        context_contrastive_loss += cont_loss
                        context_num += 1.0
            context_contrastive_loss = context_contrastive_loss/context_num
            
            ### get instance contrastive loss for slot
            
        if final_golds is not None:
            return loss_slotname, slot_contrastive_loss, context_contrastive_loss

        else:
            return pred_slotname_list


class SentRepreGenerator(nn.Module):
    def __init__(self, params, vocab):
        super(SentRepreGenerator, self).__init__()
        self.hidden_size = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        
        # LSTM Encoder for template
        self.template_encoder = Lstm(params, vocab)

        # attention layers for templates and input sequences
        self.input_atten_layer = Attention(attention_size=self.hidden_size)
        self.template_attn_layer = Attention(attention_size=self.hidden_size)
        # self.attn_layer = Attention(attention_size=self.hidden_size)

    def forward(self, templates, tem_lengths, hidden_layers, x_lengths):
        """
        Inputs:
            templates: (bsz, 3, max_template_length)
            tem_lengths: (bsz,)
            hidden_layers: (bsz, max_length, hidden_size)
            x_lengths: (bsz,)
        Outputs:
            template_sent_repre: (bsz, 3, hidden_size)
            input_sent_repre: (bsz, hidden_size)
        """
        # generate templates sentence representation
        template0 = templates[:, 0, :]
        template1 = templates[:, 1, :]
        template2 = templates[:, 2, :]

        template0_hiddens = self.template_encoder(template0)
        template1_hiddens = self.template_encoder(template1)
        template2_hiddens = self.template_encoder(template2)

        template0_repre, _ = self.template_attn_layer(template0_hiddens, tem_lengths)
        template1_repre, _ = self.template_attn_layer(template1_hiddens, tem_lengths)
        template2_repre, _ = self.template_attn_layer(template2_hiddens, tem_lengths)

        templates_repre = torch.stack((template0_repre, template1_repre, template2_repre), dim=1)  # (bsz, 3, hidden_size)

        # generate input sentence representations
        input_repre, _ = self.input_atten_layer(hidden_layers, x_lengths)

        return templates_repre, input_repre


class SLUTagger(nn.Module):
    def __init__(self, params, vocab):
        super(SLUTagger, self).__init__()

        self.lstm = Lstm(params, vocab)
        self.num_slot = params.num_slot
        self.hidden_dim = params.hidden_dim * 2 if params.bidirection else params.hidden_dim
        self.linear = nn.Linear(self.hidden_dim, self.num_slot)
        self.crf_layer = CRF(self.num_slot)
        
    def forward(self, X, lengths=None):
        """
        Input: 
            X: (bsz, seq_len)
        Output:
            prediction: (bsz, seq_len, num_slot)
            lstm_hidden: (bsz, seq_len, hidden_size)
        """
        lstm_hidden = self.lstm(X)  # (bsz, seq_len, hidden_dim)
        prediction = self.linear(lstm_hidden)

        return prediction
    
    def crf_decode(self, inputs, lengths):
        """ crf decode
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
        Ouput:
            crf_loss: loss of crf
        """
        prediction = self.crf_layer(inputs)
        prediction = [ prediction[i, :length].data.cpu().numpy() for i, length in enumerate(lengths) ]

        return prediction
    
    def crf_loss(self, inputs, lengths, y):
        """ create crf loss
        Input:
            inputs: (bsz, seq_len, num_entity)
            lengths: lengths of x (bsz, )
            y: label of slot value (bsz, seq_len)
        Ouput:
            crf_loss: loss of crf
        """
        padded_y = self.pad_label(lengths, y)
        crf_loss = self.crf_layer.loss(inputs, padded_y)

        return crf_loss

    def pad_label(self, lengths, y):
        bsz = len(lengths)
        max_len = torch.max(lengths)
        padded_y = torch.LongTensor(bsz, max_len).fill_(SLOT_PAD)
        for i in range(bsz):
            length = lengths[i]
            y_i = y[i]
            padded_y[i, 0:length] = torch.LongTensor(y_i)

        padded_y = padded_y.cuda()
        return padded_y
        

def contrastive_loss(query: torch.Tensor, keys: torch.Tensor, temperature: float = 0.1):
    """
    Given the query vector and the keys matrix (the first vector is a positive sample by default, and the rest are negative samples), calculate the contrast learning loss, follow SimCLR
    :param query: shape=(d,)
    :param keys: shape=(39, d)
    :return: scalar
    """
    query = torch.nn.functional.normalize(query, dim=0)
    keys = torch.nn.functional.normalize(keys, dim=1)
    output = torch.nn.functional.cosine_similarity(query.unsqueeze(0), keys)  # (39,)
    numerator = torch.exp(output[0] / temperature)
    denominator = torch.sum(torch.exp(output / temperature))
    return -torch.log(numerator / denominator)