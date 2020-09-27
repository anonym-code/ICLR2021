from datasets import UCI, sbm, Reddit, Elliptic, Bitcoin, AS
from model import EGNNCSp, EGNNC, Classifier
from splitter import splitter
from trainer import Trainer
from tasker import Edge_Cls_Tasker, Node_Cls_Tasker, Link_Pred_Tasker
import torch.nn as nn
import utils as u
import torch
import logger
import Cross_Entropy as ce

def build_dataset(args):
    if args.data == 'bitcoinotc' or args.data == 'bitcoinalpha':
        if args.data == 'bitcoinotc':
            args.bitcoin_args = args.bitcoinotc_args
        elif args.data == 'bitcoinalpha':
            args.bitcoin_args = args.bitcoinalpha_args
        return Bitcoin(args)
    elif args.data == 'elliptic':
        return Elliptic(args)
    elif args.data == 'UCI':
        return UCI(args)
    elif args.data == 'AS':
        return AS(args)
    elif args.data == 'reddit':
        return Reddit(args)
    elif args.data == 'sbm20':
        args.sbm_args = args.sbm20_args
        return sbm(args)
    elif args.data == 'sbm50':
        args.sbm_args = args.sbm50_args
        return sbm(args)

def build_tasker(args, dataset):
    if args.task == 'link_pred':
        return Link_Pred_Tasker(args, dataset)
    elif args.task == 'edge_cls':
        return Edge_Cls_Tasker(args, dataset)
    elif args.task == 'node_cls':
        return Node_Cls_Tasker(args, dataset)

def build_classifier(args, tasker):
    if 'node_cls' == args.task:
        mult = 1
    else:
        mult = 2
    return Classifier(args, in_features=args.gcn_out_feats * mult, out_features=tasker.num_classes).to(args.device)

if __name__ == '__main__':
    parser = u.create_parser()
    args = u.parse_args(parser)

    args.device = 'cuda'
    args.gcn_device = 'cuda'

    dataset = build_dataset(args)
    tasker = build_tasker(args, dataset)
    splitter = splitter(args, tasker)
<<<<<<< HEAD
=======
    if args.fft or args.dft:
        tim_dim = args.fft_num_steps
    else:
        tim_dim = args.num_hist_steps
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
    if args.model == 'egnncsp':
        gcn = EGNNCSp(args.feats_per_node,
                    args.hidden_feats,
                    args.gcn_out_feats,
<<<<<<< HEAD
                    args.num_hist_steps,
=======
                    tim_dim,
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
                    tasker.data.num_nodes).to(args.gcn_device)
    else:
        gcn = EGNNC(args.feats_per_node,
                    args.hidden_feats,
                    args.gcn_out_feats,
<<<<<<< HEAD
                    args.num_hist_steps,
=======
                    tim_dim,
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
                    tasker.data.num_nodes).to(args.gcn_device)
    classifier = build_classifier(args, tasker)
    loss = ce.Cross_Entropy(args, dataset).to(args.device)
    trainer = Trainer(args,
                      splitter=splitter,
                      gcn=gcn,
                      classifier=classifier,
                      comp_loss=loss,
                      dataset=dataset,
                      num_classes=tasker.num_classes)
    trainer.train()