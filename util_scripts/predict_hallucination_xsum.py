import os
import torch
import sys
from fairseq.models.roberta import RobertaModel


def make_batches(total, bsz):
    batches = []
    for ii in range(0, total, bsz):
        batches.append((ii, ii+bsz if ii + bsz < total else total))
    return batches

def convert_gpt2_labels_to_raw_labels(sent_bpe, sent_detoks, bpe_labels):
    cat_bpe = sent_bpe.split()
    assert len(cat_bpe) == len(bpe_labels)
    detok_labels = []
    detoks = []

    atom = []
    labels = []
    for token, label in zip(cat_bpe, bpe_labels):
        debped_token = roberta.bpe.decode(token)
        # print(token, debped_token)
        # input()
        if len(atom) == 0:
            atom.append(debped_token)
            labels.append(label)
        elif debped_token.startswith(' ') and len(atom) > 0:
            detok_labels.append(1 if sum(labels) > 0 else 0)
            recover = "".join(atom).strip()
            detoks.append(recover)

            atom = []
            atom.append(debped_token)
            labels = []
            labels.append(label)
        else:
            atom.append(debped_token)
            labels.append(label)

    if len(atom) > 0 and len(labels) > 0:
        detok_labels.append(1 if sum(labels) > 0 else 0)
        token = "".join(atom).strip()
        detoks.append(token)
    assert len(sent_detoks) == len(detok_labels)
    # assert " ".join(detoks) == " ".join(sent_detoks)
    return detok_labels

if __name__ == '__main__':
    suffix = sys.argv[1]
    raw_dir = "data"
    source_fname = "source"
    hypo_fname = "hypo" + suffix

    model_path = "models/xsum.roberta.tar.gz"
    datapath = "models/xsum.roberta.tar.gz/data"
    opt_dir = "logs"
    if not os.path.exists(opt_dir):
        os.mkdir(opt_dir)
    print("log dir: " + opt_dir)
    flog = open(os.path.join(opt_dir, "hal_pred" + suffix + ".log"), "w", encoding="utf-8")
    flabel = open(os.path.join(opt_dir, "label" + suffix), "w", encoding="utf-8")

    # read input in
    data = []
    with open(os.path.join(raw_dir, source_fname), encoding='utf-8') as fin1, \
            open(os.path.join(raw_dir, hypo_fname), encoding='utf-8') as fin2:
        for l1, l2 in zip(fin1, fin2):
            data.append((l1.strip(), l2.strip()))

    # """
    print(model_path)
    roberta = RobertaModel.from_pretrained(
        model_path,
        checkpoint_file='checkpoint.pt',
        data_name_or_path=datapath
    )
    raw = True
    print("Loaded the model!")

    roberta.cuda()
    roberta.eval()
    roberta.half()

    max_positions = roberta.model.max_positions()
    # """
    max_positions = 512
    use_ref = 0
    print(f"use ref = {use_ref}")

    bsz = 100 # batch size

    for i, j in make_batches(len(data), bsz):
        slines = [[sample[0] for sample in data[i: j]], [sample[1] for sample in data[i: j]]]
        first_seg_lengths = None
        # if raw, target are detoknized labels
        with torch.no_grad():
            src, tgt = slines[0], slines[1]
            prediction_label, prediction_probs, target_bpes = roberta.predict_hallucination_labels(src, tgt,
                                                                first_seg_lengths=first_seg_lengths,
                                                                raw=True,
                                                                inputs_ref=None)
        # convert bpe labels to raw labels
        full_bpes = [bpe for sent in target_bpes for bpe in sent.split()]
        assert len(full_bpes) == len(prediction_label)
        cum_lengths = 0
        for idx, (raw_target, sent) in enumerate(zip(slines[1], target_bpes)):
            token_prediction_labels = convert_gpt2_labels_to_raw_labels(sent,
                                                                        raw_target.split(),
                                                                        prediction_label[
                                                                        cum_lengths:cum_lengths + len(
                                                                            sent.split())])
            cum_lengths += len(sent.split())
            flog.write("Token-Prediction: " + " ".join(["{}[{}]".format(t, p) for t, p in zip(raw_target.split(), token_prediction_labels)]) + "\n")
            flabel.write(" ".join(["{}".format(p) for p in token_prediction_labels]) + "\n")
