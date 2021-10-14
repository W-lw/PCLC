#!/usr/bin/env python

# Python version of the evaluation script from CoNLL'00-

# Intentional differences:
# - accept any space as delimiter by default
# - optional file argument (default STDIN)
# - option to set boundary (-b argument)
# - LaTeX output (-l argument) not supported
# - raw tags (-r argument) not supported

import sys
import re

from collections import defaultdict, namedtuple

ANY_SPACE = '<SPACE>'

class FormatError(Exception):
    pass

Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')

from preprocess.gen_embeddings_for_slu import unseen_slot,seen_slot

class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0    # number of correctly identified chunks
        self.correct_tags = 0     # number of correct chunk tags
        self.found_correct = 0    # number of chunks in corpus
        self.found_guessed = 0    # number of identified chunks  
        self.token_counter = 0    # token counter (ignores sentence breaks)


        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)
        self.t_found_entity = defaultdict(int)
        self.t_found_transition  = defaultdict(lambda: defaultdict(int)) # the slot distribution predicted by the model

        self.correct_chunk_list = []
        self.guessed_chunk_list = []

def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser(
        description='evaluate tagging results using CoNLL criteria',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)

def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluate(lines, options=None):
    if options is None:
        options = parse_args([])    # use defaults

    counts = EvalCounts()
    num_features = None       # number of features per line
    in_correct = False        # currently processed chunks is correct until now\
    span_start = False        # check whether the start of the span is correct
    last_correct = 'O'        # previous chunk tag in corpus
    last_correct_type = ''    # type of previously identified chunk tag
    last_guessed = 'O'        # previously identified chunk tag
    last_guessed_type = ''    # type of previous chunk tag in corpus

    for line in lines:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)
        # label and predicted value of each token, including BIO and slot
        correct, correct_type = parse_tag(features.pop())
        guessed, guessed_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif (end_correct != end_guessed or guessed_type != correct_type):
                in_correct = False

        if span_start:
            """
            It is only used to observe how the predicted slot is transferred,
            and does not participate in the f1 score calculation.
            """
            if(end_correct and end_guessed and last_correct_type!=""):
                counts.t_found_entity[last_correct_type] +=1
                if last_correct_type != last_guessed_type:
                    counts.t_found_transition[last_correct_type][last_guessed_type] += 1
                span_start = False
                counts.correct_chunk_list.append(last_correct_type)
                counts.guessed_chunk_list.append(last_guessed_type)

            elif(end_correct!= end_guessed):
                span_start = False

        if start_correct and start_guessed:
            span_start =  True 
            if guessed_type == correct_type:
                in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:          
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts

def uniq(iterable):
  seen = set()
  return [i for i in iterable if not (i in seen or seen.add(i))]

def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed-correct, total-correct
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)

def calcualate_f1_score(tp, fp, fn):
    p = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return f

def metrics(counts):
    """
    Perform statistics on all seen slots and unseen slots, and recalculate the final f1 score on seen slots and unseen slots.
    When we calculate the f1 score, we only base it on whether the slot is seen or unseen, not based on the sentence.
    """
    
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    unseen_tp = 0
    unseen_fp = 0
    unseen_fn = 0
    unseen_count = 0
    seen_tp = 0
    seen_fp = 0
    seen_fn = 0
    seen_count = 0
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    # for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
    #     # print(t)
    #     if t not in c.t_found_transition.keys():
    #         print("no error samples")
    #     else:
    #         print(c.t_found_transition[t])
    #         continue
    BIO_tag = True  # check if at the BIO stage
    for slots in by_type.keys():
        p = by_type[slots]
        if slots in unseen_slot:
            unseen_count += 1
            unseen_tp += getattr(p,"tp")
            unseen_fp += getattr(p,"fp")
            unseen_fn += getattr(p,"fn")
            
        elif slots in seen_slot:
            seen_count +=1
            seen_tp += getattr(p,"tp")
            seen_fp += getattr(p,"fp")
            seen_fn += getattr(p,"fn")
        else:
            BIO_tag = False
            print("BIO stage:")

    if BIO_tag and unseen_count == 0:
        print("no unseen slot in this domain")
        final_unseen_f1_score = 0
    else:        
        final_unseen_f1_score = calcualate_f1_score(unseen_tp,unseen_fp,unseen_fn)
    
    if BIO_tag and seen_count == 0:
        print("no seen slot in this domain")
        final_seen_f1_score = 0
    else:       
        final_seen_f1_score = calcualate_f1_score(seen_tp,seen_fp,seen_fn)
    
    if BIO_tag:
        print("final unseen f1 score:"+str(final_unseen_f1_score*100))
        print("final seen f1 score:"+str(final_seen_f1_score*100))

    return overall, by_type


from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def report_confusion_matrix(counts:EvalCounts,params):
    all_labels = list(set(counts.correct_chunk_list)|set(counts.guessed_chunk_list))
    all_labels.sort()
    labels2idx = {x:i for i, x in enumerate(all_labels)}
    predicts = np.array([labels2idx[x] for x in counts.guessed_chunk_list])
    golds = np.array([labels2idx[x] for x in counts.correct_chunk_list])
    
    np.set_printoptions(threshold=np.inf)
    cm = confusion_matrix(golds, predicts)
    plot_confusion_matrix(cm, list(labels2idx.keys()), "./test_image/transfer/res_"+params.tgt_dm+".png", True)

import itertools

def plot_confusion_matrix(cm,classes, savename, normalize=False, title=f'Confusion Matrix', figsize=(12,9), cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
    np.set_printoptions(precision=2)
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)# align="center")
    plt.yticks(tick_marks, classes)
    plt.ylabel("true")
    plt.xlabel("pred")
    plt.savefig(savename)


def report(counts, params,out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)
    #report_confusion_matrix(counts,params)  # drawing the confusion matrix
    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    results = {}
    if c.token_counter > 0:
        results["fb1"] = 100.*overall.fscore
    print(by_type)
    return results

def end_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk ended between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    # these chunks are assumed to have length 1
    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end

def  start_of_chunk(prev_tag, tag, prev_type, type_):
    # check if a chunk started between the previous and current word
    # arguments: previous and current chunk tags, previous and current types
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start

def main(argv):
    args = parse_args(argv[1:])

    if args.file is None:
        counts = evaluate(sys.stdin, args)
    else:
        with open(args.file) as f:
            counts = evaluate(f, args)
    report(counts)

def conll2002_measure(params,lines, verbose=False):
    counts = evaluate(lines, None)
    return report(counts,params)