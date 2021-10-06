
from src.utils import init_experiment
from src.slu.datareader import datareader, read_file, binarize_data
from src.slu.dataloader import get_dataloader, Dataset, DataLoader, collate_fn
from src.slu.baseline_loader import get_dataloader as get_baselineloader
from src.slu.baseline_loader import collate_fn as baseline_collate_fn
from src.slu.baseline_loader import Dataset as BaselineDataset
from src.slu.trainer import SLUTrainer
from src.slu.baseline_trainer import BaselineTrainer
from preprocess.gen_embeddings_for_slu import domain2slot
from config import get_params
import numpy as np
import torch
import os

from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.lib.function_base import select
from sklearn.decomposition import PCA
from sklearn import manifold
from preprocess.gen_embeddings_for_slu import domain2slot
import pdb
import pickle
from datetime import datetime

slot_name = ["AddToPlaylist","BookRestaurant","GetWeather","PlayMusic","RateBook","SearchCreativeWork","SearchScreeningEvent"]

def test_coach(params):
    # get dataloader
    _, _, dataloader_test, _ = get_dataloader(params.tgt_dm, params.batch_size, params.tr, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)
    
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    singal_domain_slots_embedding = {}
    all_slots_embedding = {}

    #target domain singal_image
    singal_embs = torch.FloatTensor(slotname_predictor.slot_embs[params.tgt_dm]).cuda()
    singal_embs = slotname_predictor.slot_name_projection_for_slot(singal_embs)
    singal_embs = singal_embs.cpu().detach().numpy()
    for idx, slot_embedding in enumerate(singal_embs):
        singal_domain_slots_embedding[domain2slot[params.tgt_dm][idx]] = slot_embedding 
    plotTsneForSlotName(singal_domain_slots_embedding,params.tgt_dm,params.model_path)


    #target domain all_image
    for domains in slot_name:
        embs = torch.FloatTensor(slotname_predictor.slot_embs[domains]).cuda()
        embs = slotname_predictor.slot_name_projection_for_slot(embs)
        embs = embs.cpu().detach().numpy()
        for idx, slot_embedding in enumerate(embs):
            all_slots_embedding[domain2slot[domains][idx]] = slot_embedding
    
    plotTsneForSlotName(all_slots_embedding,params.tgt_dm,params.model_path,a="all")

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)

    bio_score, f1_score, _ = slu_trainer.evaluate(params,dataloader_test, istestset=True,test_tag=False)
    print(f"target domain:{params.tgt_dm}")
    dt = datetime.now()
    print(dt.strftime("%Y-%m-%d %H:%M:%S"))
    print("Eval on test set.BIO Slot F1 Score: {:.4f}, Final Slot F1 Score: {:.4f}.".format(bio_score,f1_score))
    print("------------------------------------------------------------")


def test_baseline(params):
    # get dataloader
    _, _, dataloader_test, _ = get_baselineloader(params.tgt_dm, params.batch_size, params.n_samples)

    model_path = params.model_path
    assert os.path.isfile(model_path)

    reloaded = torch.load(model_path)
    slu_tagger = reloaded["slu_tagger"]
    slu_tagger.cuda()

    baseline_trainer = BaselineTrainer(params, slu_tagger)

    _, f1_score, _ = baseline_trainer.evaluate(0, dataloader_test, istestset=True)
    print("Eval on test set. Slot F1 Score: {:.4f}.".format(f1_score))


def test_coach_on_seen_and_unseen(params):
    # read seen and unseen data
    print(params.tgt_dm+" "+params.exp_id)
    print("Getting vocabulary ...")
    _, vocab = datareader(params.tr)

    print("Processing Unseen and Seen samples in %s domain ..." % params.tgt_dm)
    unseen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/unseen_slots.txt", vocab, False)
    seen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/seen_slots.txt", vocab, False)

    print("Binarizing data ...")
    if len(unseen_data["utter"]) > 0:
        unseen_data_bin = binarize_data(unseen_data, vocab, params.tgt_dm, False)
    else:
        unseen_data_bin = None
    
    if len(seen_data["utter"]) > 0:
        seen_data_bin = binarize_data(seen_data, vocab, params.tgt_dm, False)
    else:
        seen_data_bin = None

    model_path = params.model_path
    assert os.path.isfile(model_path)
    reloaded = torch.load(model_path)
    binary_slu_tagger = reloaded["binary_slu_tagger"]
    slotname_predictor = reloaded["slotname_predictor"]
    binary_slu_tagger.cuda()
    slotname_predictor.cuda()

    slu_trainer = SLUTrainer(params, binary_slu_tagger, slotname_predictor)

    print("Prepare dataloader ...")
    if unseen_data_bin and params.unseen_seen == "unseen":
        unseen_dataset = Dataset(unseen_data_bin["utter"], unseen_data_bin["y1"], unseen_data_bin["y2"], unseen_data_bin["domains"])
        unseen_dataloader = DataLoader(dataset=unseen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)
        _,f1_score, _ = slu_trainer.evaluate(params,unseen_dataloader, istestset=True)
        print("Evaluate on {} domain unseen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of unseen sample is zero")

    if seen_data_bin and params.unseen_seen == "seen":
        seen_dataset = Dataset(seen_data_bin["utter"], seen_data_bin["y1"], seen_data_bin["y2"], seen_data_bin["domains"])
        seen_dataloader = DataLoader(dataset=seen_dataset, batch_size=params.batch_size, collate_fn=collate_fn, shuffle=False)
        _,f1_score, _ = slu_trainer.evaluate(params,seen_dataloader, istestset=True)
        print("Evaluate on {} domain seen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))
        print("-------------------------------------------------------")

    else:
        print("Number of seen sample is zero")
        print("-------------------------------------------------------")


def test_baseline_on_seen_and_unseen(params):
    # read seen and unseen data
    print("Getting vocabulary ...")
    _, vocab = datareader()

    print("Processing Unseen and Seen samples in %s domain ..." % params.tgt_dm)
    unseen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/unseen_slots.txt", vocab, False)
    seen_data, vocab = read_file("data/snips/"+params.tgt_dm+"/seen_slots.txt", vocab, False)

    print("Binarizing data ...")
    if len(unseen_data["utter"]) > 0:
        unseen_data_bin = binarize_data(unseen_data, vocab, params.tgt_dm, False)
    else:
        unseen_data_bin = None
    
    if len(seen_data["utter"]) > 0:
        seen_data_bin = binarize_data(seen_data, vocab, params.tgt_dm, False)
    else:
        seen_data_bin = None

    model_path = params.model_path
    assert os.path.isfile(model_path)

    reloaded = torch.load(model_path)
    slu_tagger = reloaded["slu_tagger"]
    slu_tagger.cuda()

    baseline_trainer = BaselineTrainer(params, slu_tagger)

    print("Prepare dataloader ...")
    if unseen_data_bin:
        unseen_dataset = BaselineDataset(unseen_data_bin["utter"], unseen_data_bin["y2"], unseen_data_bin["domains"])

        unseen_dataloader = DataLoader(dataset=unseen_dataset, batch_size=params.batch_size, collate_fn=baseline_collate_fn, shuffle=False)

        _, f1_score, _ = baseline_trainer.evaluate(0, unseen_dataloader, istestset=True)
        print("Evaluate on {} domain unseen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of unseen sample is zero")

    if seen_data_bin:
        seen_dataset = BaselineDataset(seen_data_bin["utter"], seen_data_bin["y2"], seen_data_bin["domains"])
        
        seen_dataloader = DataLoader(dataset=seen_dataset, batch_size=params.batch_size, collate_fn=baseline_collate_fn, shuffle=False)

        _, f1_score, _ = baseline_trainer.evaluate(0, seen_dataloader, istestset=True)
        print("Evaluate on {} domain seen slots. Final slot F1 score: {:.4f}.".format(params.tgt_dm, f1_score))

    else:
        print("Number of seen sample is zero")

def plotTsneForSlotName(embs_dict,tgt_dm,model_path, a=None):
    x = np.stack(embs_dict.values())
    labels = list(embs_dict.keys())
    def cos_similarity(x,y):
            num = x.dot(y.T)
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            return (num / denom)
    tsne = manifold.TSNE(n_components=2, init='pca',perplexity=30, n_iter=10000,early_exaggeration=12, random_state=1
    # metric=cos_similarity
    )
    red_features = tsne.fit_transform(x)
    
    # if a == 'all':
    #     plot_with_labels(red_features,labels,tgt_dm,model_path,a='all')
    # else:
    #     plot_with_labels(red_features,labels,tgt_dm,model_path)

slot_list = ['playlist', 'music_item', 'geographic_poi', 'facility', 'movie_name', 'location_name', 'restaurant_name', 'track', 'restaurant_type', 'object_part_of_series_type', 'country', 'service', 'poi', 'party_size_description', 'served_dish', 'genre', 'current_location', 'object_select', 'album', 'object_name', 'state', 'sort', 'object_location_type', 'movie_type', 'spatial_relation', 'artist', 'cuisine', 'entity_name', 'object_type', 'playlist_owner', 'timeRange', 'city', 'rating_value', 'best_rating', 'rating_unit', 'year', 'party_size_number', 'condition_description', 'condition_temperature']

domain2slot = {
    "AddToPlaylist": ['music_item', 'playlist_owner', 'entity_name', 'playlist', 'artist'],
    "BookRestaurant": ['city', 'facility', 'timeRange', 'restaurant_name', 'country', 'cuisine', 'restaurant_type', 'served_dish', 'party_size_number', 'poi', 'sort', 'spatial_relation', 'state', 'party_size_description'],
    "GetWeather": ['city', 'state', 'timeRange', 'current_location', 'country', 'spatial_relation', 'geographic_poi', 'condition_temperature', 'condition_description'],
    "PlayMusic": ['genre', 'music_item', 'service', 'year', 'playlist', 'album','sort', 'track', 'artist'],
    "RateBook": ['object_part_of_series_type', 'object_select', 'rating_value', 'object_name', 'object_type', 'rating_unit', 'best_rating'],
    "SearchCreativeWork": ['object_name', 'object_type'],
    "SearchScreeningEvent": ['timeRange', 'movie_type', 'object_location_type','object_type', 'location_name', 'spatial_relation', 'movie_name']
}

def plot_with_labels(features, labels,tgt_dm,model_path,a=None):

    plt.cla()
    plt.style.use("seaborn-darkgrid")
    X, Y = features[:,0], features[:,1]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    if a == 'all':
        B = slot_name
        B.remove(tgt_dm)
        print(B)
        A = set(domain2slot[tgt_dm])
        print('A:')
        print(A)   
        D = set()
        for name in B:
            Z = set(domain2slot[name])
            D = D.union(Z)
        # find the unique target domain slot
        duyou = A.difference(D)
        gongyou = A.intersection(D) 
        print(duyou)
        print(gongyou) 
    else:
        B = slot_name

    for x, y, s,i in zip(X, Y, labels, range(len(labels))):
        if a == 'all':    
            if s in duyou:
                c = '#17D1F9'
                ec = '#048BA8'
                ecc = '#A2EDFD'
                x = x * 0.8
                y = y * 0.8
                ax.scatter(x, y, c = c, s = 300, edgecolors = ec , linewidths= 2 )
                ax.scatter(x, y, c = ecc, s = 50)
                ax.text(x+5, y+5, s, c = 'b')
            elif s in gongyou:
                c = '#A849E7'
                ec = '#66149C'
                ecc = '#C07BEE'
                ax.scatter(x, y, c = c, s = 300, edgecolors = ec , linewidths= 2 ) 
                ax.scatter(x, y, c = ecc, s = 50)
                ax.text(x+5, y+5, s, c = '#800080') 
            else:
                c = '#FB3E15'
                ec = '#A52003'
                ecc = '#FD9680'
                x = x * 1.2
                y = y * 1.2
                ax.scatter(x, y, c = c, s = 300, edgecolors = ec , linewidths= 2 )
                ax.scatter(x, y, c = ecc, s = 50)
                ax.text(x+5, y+5, s, c = 'r') 
        else: 
            c = 'b'
            ax.scatter(x, y, c = c, s = 300)    
    ax.set_xlim(X.min() * 1.5, X.max() * 1.5)
    ax.set_ylim(Y.min() * 1.5, Y.max() * 1.5)
    ax.set_title('Visualize last layer', pad=15, fontsize=20)
    ax.grid(True)
    x = "Visualization/slot_embedding/"
    y = "Visualization/projection_slot_embedding/"
    z = model_path.replace("best_model.pth","")
    l = "test_image/tsne/"
    m = "Visualization/test_50/"

    if a == 'all':
        plt.savefig(l +'all_'+tgt_dm+".png",dpi=700)
    else:
        plt.savefig(l +tgt_dm+".png")

if __name__ == "__main__":
    params = get_params()
    if params.model_type == "coach":
        if params.test_mode == "testset":
            test_coach(params)
        elif params.test_mode == "seen_unseen":
            test_coach_on_seen_and_unseen(params)
    else:
        if params.test_mode == "testset":
            test_baseline(params)
        elif params.test_mode == "seen_unseen":
            test_baseline_on_seen_and_unseen(params)
