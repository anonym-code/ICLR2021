import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
import os

class Trainer:
    def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier
        self.comp_loss = comp_loss

        self.num_nodes = dataset.num_nodes
        self.data = dataset
        self.num_classes = num_classes

        self.logger = logger.Logger(args, self.num_classes)

        self.init_optimizers(args)


    def init_optimizers(self, args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()


    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)


    def load_checkpoint(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(checkpoint['classifier_optimizer'])
            self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
            return epoch
        else:
            self.logger.log_str("=> no checkpoint found at '{}'".format(filename))
            return 0

    def train(self):
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        for e in range(self.args.num_epochs):
            eval_train, nodes_embs = self.run_epoch(self.splitter.train, e, 'TRAIN', grad=True)
            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                eval_valid, _ = self.run_epoch(self.splitter.dev, e, 'VALID', grad=False)
                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    print('### w' + ') ep ' + str(e) + ' - Best valid measure:' + str(eval_valid))
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print('### w' + ') ep ' + str(e) + ' - Early stop.')
                        break

            if len(self.splitter.test) > 0 and e > self.args.eval_after_epochs:
                eval_test, _ = self.run_epoch(self.splitter.test, e, 'TEST', grad=False)

                if self.args.save_node_embeddings:
                    self.save_node_embs_csv(nodes_embs, self.splitter.train_idx, log_file + '_train_nodeembs.csv.gz')
                    self.save_node_embs_csv(nodes_embs, self.splitter.dev_idx, log_file + '_valid_nodeembs.csv.gz')
                    self.save_node_embs_csv(nodes_embs, self.splitter.test_idx, log_file + '_test_nodeembs.csv.gz')

    def run_epoch(self, split, epoch, set_name, grad):
        t0 = time.time()
        log_interval = 999
        if set_name == 'TEST':
            log_interval = 1
        self.logger.log_epoch_start(epoch, len(split), set_name, minibatch_log_interval=log_interval)

        torch.set_grad_enabled(grad)
        for s in split:
            s = self.prepare_sample(s)
            predictions, nodes_embs = self.predict(s.edge_feature,
                                                   s.label_sp['idx'],
                                                   s.node_feature,
                                                   set_name)
            #print((s.label_sp['vals']==1).sum(), s.label_sp['vals'].size())
            loss = self.comp_loss(predictions, s.label_sp['vals'])
            if set_name in ['TEST', 'VALID'] and self.args.task == 'link_pred':
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach(), adj=s.label_sp['idx'])
            else:
                self.logger.log_minibatch(predictions, s.label_sp['vals'], loss.detach())
            if grad:
                self.optim_step(loss)

        torch.set_grad_enabled(True)
        eval_measure = self.logger.log_epoch_done()

        return eval_measure, nodes_embs


    def gather_node_embs(self, nodes_embs, node_indices):
        cls_input = []
        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input, dim=1)


    def predict(self, edge_feature, node_indices, node_feature, set_name):
        if set_name == 'TEST' or 'VALID':
            self.gcn.eval()
        else:
            self.gcn.train()
        if node_feature == 1:
            nodes_embs = self.gcn(edge_feature.to(self.args.device), torch.arange(self.tasker.data.num_nodes).to(self.args.device))
        else:
            nodes_embs = self.gcn(edge_feature.to(self.args.device), torch.arange(self.tasker.data.num_nodes).to(self.args.device),
                                  node_feature.to(self.args.device))
        predict_batch_size = 100000
        gather_predictions = []
        for i in range(1 + (node_indices.size(1) // predict_batch_size)):
            cls_input = self.gather_node_embs(nodes_embs,
                                              node_indices[:,  i*predict_batch_size:(i+1)*predict_batch_size])
            predictions = self.classifier(cls_input)
            gather_predictions.append(predictions)
        gather_predictions = torch.cat(gather_predictions, dim=0)
        return gather_predictions, nodes_embs


    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

    def prepare_sample(self, sample):
        sample = u.Namespace(sample)
        sample.edge_feature = sample.edge_feature.squeeze()
        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            label_sp['idx'] = label_sp['idx'].to(
                self.args.device).t()
        else:
            label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(torch.cat((orig_ID, nodes_embs[node_id].double())).detach().numpy())

        pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
