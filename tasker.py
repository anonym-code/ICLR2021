import torch
import taskers_utils as tu
import utils as u
from scipy.fftpack import dct, idctn
import numpy as np

class Edge_Cls_Tasker:
    def __init__(self, args, dataset):
        self.data = dataset
        # max_time for link pred should be one before
        self.max_time = dataset.max_time
        self.args = args
        self.num_classes = dataset.num_classes


    def get_sample(self, idx, test):
        hist_adj_list = []
        hist_mask_list = []

        for i in range(idx - self.args.num_hist_steps+1, idx + 1):
            cur_adj = tu.get_edge_labels(edges=self.data.edges,
                                        time=i)
            #cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)
            cur_adj = torch.sparse.LongTensor(cur_adj['idx'].T, cur_adj['vals'])
            hist_adj_list.append(cur_adj)

        # This would be if we were training on all the edges in the time_window
        label_adj = tu.get_edge_labels(edges = self.data.edges, 
								  	   time = idx)

        return {'idx': idx,
                'edge_feature': hist_adj_list,
                'label_sp': label_adj,
                'node_feature': 1}

class Node_Cls_Tasker:
    def __init__(self,args,dataset):
        self.data = dataset

        self.max_time = dataset.max_time

        self.args = args

        self.num_classes = 2

        self.feats_per_node = dataset.feats_per_node

        self.nodes_labels_times = dataset.nodes_labels_times




    def get_sample(self,idx,test):
        hist_adj_list = []
        hist_mask_list = []
<<<<<<< HEAD
        node_feature = 1
        for i in range(idx - self.args.num_hist_steps, idx+1):
=======
        node_feature = torch.FloatTensor(self.data.nodes_feats)
        for i in range(idx - self.args.num_hist_steps+1, idx+1):
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
            #all edgess included from the beginning
            cur_adj = tu.get_sp_adj(edges = self.data.edges,
                                    time = i,
                                    weighted = True,
                                    time_window = self.args.adj_mat_time_window) #changed this to keep only a time window

<<<<<<< HEAD
            cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)
            cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].T, cur_adj['vals']).to_dense()

=======
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
            hist_adj_list.append(cur_adj)

        label_adj = self.get_node_labels(idx)

        return {'idx': idx,
                'edge_feature': hist_adj_list,
                'label_sp': label_adj,
                'node_feature': node_feature}


    def get_node_labels(self,idx):
        node_labels = self.nodes_labels_times
        subset = node_labels[:,2]==idx
        label_idx = node_labels[subset,0]
        label_vals = node_labels[subset,1]

        return {'idx': label_idx,
                'vals': label_vals}


class Link_Pred_Tasker:
    '''
    Creates a tasker object which computes the required inputs for training on a link prediction
    task. It receives a dataset object which should have two attributes: nodes_feats and edges, this
    makes the tasker independent of the dataset being used (as long as mentioned attributes have the same
    structure).

    Based on the dataset it implements the get_sample function required by edge_cls_trainer.
    This is a dictionary with:
        - time_step: the time_step of the prediction
        - hist_adj_list: the input adjacency matrices until t, each element of the list
                         is a sparse tensor with the current edges. For link_pred they're
                         unweighted
        - nodes_feats_list: the input nodes for the GCN models, each element of the list is a tensor
                          two dimmensions: node_idx and node_feats
        - label_adj: a sparse representation of the target edges. A dict with two keys: idx: M by 2
                     matrix with the indices of the nodes conforming each edge, vals: 1 if the node exists
                     , 0 if it doesn't

    There's a test difference in the behavior, on test (or development), the number of sampled non existing
    edges should be higher.
    '''

    def __init__(self, args, dataset):
        self.data = dataset
        # max_time for link pred should be one before
        self.max_time = dataset.max_time - 1
        self.args = args
        self.num_classes = 2

        self.is_static = False


    def get_sample(self, idx, test, **kwargs):
        hist_adj_list = []
        existing_nodes = []
        if self.args.model == 'egnnc':
            if self.args.fft:
                for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
                    cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                            time=i,
                                            weighted=True,
                                            time_window=self.args.adj_mat_time_window)
                    if self.args.smart_neg_sampling:
                        existing_nodes.append(cur_adj['idx'].unique())
                    else:
                        existing_nodes = None

                    # node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)
<<<<<<< HEAD

                    cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)
                    cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].T, cur_adj['vals']).to_dense().numpy()
                    hist_adj_list.append(cur_adj)
                hist_adj_list = np.concatenate(hist_adj_list).reshape((-1, cur_adj.shape[0], cur_adj.shape[1]))
                #print(1, hist_adj_list.shape)
                f_adj = dctn(hist_adj_list, axes=0, norm='ortho')
                edge_feature = torch.from_numpy(f_adj[:self.args.num_hist_steps, :, :])
                #print(2, edge_feature.size())

=======
                    #cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)

                    cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].t(),cur_adj['vals'].type(torch.float),torch.Size([self.data.num_nodes,self.data.num_nodes])).to_dense()
                    hist_adj_list.append(cur_adj)
                    
                hist_adj_list = torch.cat(hist_adj_list).view(-1,self.data.num_nodes,self.data.num_nodes,-1).numpy()
                #print(1, hist_adj_list.shape)
                edge_feature = u.DTFT(hist_adj_list, self.args.fft_num_steps)
                #print(2, edge_feature.size())
            elif self.args.dft:
                for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
                    cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                            time=i,
                                            weighted=True,
                                            time_window=self.args.adj_mat_time_window)
                    if self.args.smart_neg_sampling:
                        existing_nodes.append(cur_adj['idx'].unique())
                    else:
                        existing_nodes = None

                    # node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

                    cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)
                    cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].T, cur_adj['vals']).to_dense()
                    hist_adj_list.append(cur_adj)
                    
                hist_adj_list = torch.cat(hist_adj_list).view(-1,self.data.num_nodes,self.data.num_nodes).numpy()
                #print(1, hist_adj_list.shape)
                edge_feature = torch.from_numpy(dct(hist_adj_list, n=self.args.fft_num_steps, axis=0, norm='ortho'))
                #print(2, edge_feature.size())
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
            else:
                for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
                    cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                            time=i,
                                            weighted=True,
                                            time_window=self.args.adj_mat_time_window)
                    if self.args.smart_neg_sampling:
                        existing_nodes.append(cur_adj['idx'].unique())
                    else:
                        existing_nodes = None

                    #node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

                    cur_adj = tu.normalize_adj(adj=cur_adj, num_nodes=self.data.num_nodes)
                    cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].T, cur_adj['vals']).to_dense()
                    hist_adj_list.append(cur_adj)

<<<<<<< HEAD
                edge_feature = torch.cat(hist_adj_list).view(-1, cur_adj.size(0), cur_adj.size(1))
            concate_adj = torch.sum(edge_feature, dim=0)
            edge_feature = edge_feature.permute(1, 2, 0)
            concate_adj[concate_adj > 0] = 1
=======
                edge_feature = torch.cat(hist_adj_list).view(-1,self.data.num_nodes,self.data.num_nodes)
            edge_feature = edge_feature.permute(1, 2, 0)
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e

            # This would be if we were training on all the edges in the time_window
            label_adj = tu.get_sp_adj(edges=self.data.edges,
                                    time=idx + 1,
                                    weighted=False,
                                    time_window=self.args.adj_mat_time_window)
            if test:
                neg_mult = self.args.negative_mult_test
            else:
                neg_mult = self.args.negative_mult_training

            if self.args.smart_neg_sampling:
                existing_nodes = torch.cat(existing_nodes)

            if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
                non_exisiting_adj = tu.get_all_non_existing_edges(adj=label_adj, tot_nodes=self.data.num_nodes)
            else:
                non_exisiting_adj = tu.get_non_existing_edges(adj=label_adj,
                                                            number=label_adj['vals'].size(0) * neg_mult,
                                                            tot_nodes=self.data.num_nodes,
                                                            smart_sampling=self.args.smart_neg_sampling,
                                                            existing_nodes=existing_nodes)


            label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
            label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
            return {'idx': idx,
<<<<<<< HEAD
                    'concate_adj': concate_adj,
=======
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
                    'edge_feature': edge_feature,
                    'label_sp': label_adj,
                    'node_feature': 1}

        else:
<<<<<<< HEAD
            for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
=======
            if self.args.fft:
                for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e
                    cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                            time=i,
                                            weighted=True,
                                            time_window=self.args.adj_mat_time_window)
                    if self.args.smart_neg_sampling:
                        existing_nodes.append(cur_adj['idx'].unique())
                    else:
                        existing_nodes = None

<<<<<<< HEAD
                    #node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

                    # cur_adj = {'idx':cur_adj['idx'], 'val':cur_adj['val']}
                    hist_adj_list.append(cur_adj)

            # This would be if we were training on all the edges in the time_window
            label_adj = tu.get_sp_adj(edges=self.data.edges,
                                    time=idx + 1,
                                    weighted=False,
                                    time_window=self.args.adj_mat_time_window)
            if test:
                neg_mult = self.args.negative_mult_test
            else:
                neg_mult = self.args.negative_mult_training

            if self.args.smart_neg_sampling:
                existing_nodes = torch.cat(existing_nodes)

            if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
                non_exisiting_adj = tu.get_all_non_existing_edges(adj=label_adj, tot_nodes=self.data.num_nodes)
            else:
                non_exisiting_adj = tu.get_non_existing_edges(adj=label_adj,
                                                            number=label_adj['vals'].size(0) * neg_mult,
                                                            tot_nodes=self.data.num_nodes,
                                                            smart_sampling=self.args.smart_neg_sampling,
                                                            existing_nodes=existing_nodes)


            label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
            label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
            return {'idx': idx,
                    'edge_feature': hist_adj_list,
                    'label_sp': label_adj,
                    'node_feature': 1}
=======
                    # node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

                    cur_adj = torch.sparse.FloatTensor(cur_adj['idx'].t(),cur_adj['vals'].type(torch.float),torch.Size([self.data.num_nodes,self.data.num_nodes])).to_dense()
                    hist_adj_list.append(cur_adj)
                hist_adj_list = torch.cat(hist_adj_list).view(self.data.num_nodes,self.data.num_nodes,-1).numpy()
                #print(1, hist_adj_list.shape)
                fft_hist_adj_list = u.DTFTSp(hist_adj_list, self.args.fft_num_steps)

                # This would be if we were training on all the edges in the time_window
                label_adj = tu.get_sp_adj(edges=self.data.edges,
                                        time=idx + 1,
                                        weighted=False,
                                        time_window=self.args.adj_mat_time_window)
                if test:
                    neg_mult = self.args.negative_mult_test
                else:
                    neg_mult = self.args.negative_mult_training

                if self.args.smart_neg_sampling:
                    existing_nodes = torch.cat(existing_nodes)

                if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
                    non_exisiting_adj = tu.get_all_non_existing_edges(adj=label_adj, tot_nodes=self.data.num_nodes)
                else:
                    non_exisiting_adj = tu.get_non_existing_edges(adj=label_adj,
                                                                number=label_adj['vals'].size(0) * neg_mult,
                                                                tot_nodes=self.data.num_nodes,
                                                                smart_sampling=self.args.smart_neg_sampling,
                                                                existing_nodes=existing_nodes)


                label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
                label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
                return {'idx': idx,
                        'edge_feature': fft_hist_adj_list,
                        'label_sp': label_adj,
                        'node_feature': 1}
            else:
                for i in range(idx - self.args.num_hist_steps + 1, idx + 1):
                        cur_adj = tu.get_sp_adj(edges=self.data.edges,
                                                time=i,
                                                weighted=True,
                                                time_window=self.args.adj_mat_time_window)
                        if self.args.smart_neg_sampling:
                            existing_nodes.append(cur_adj['idx'].unique())
                        else:
                            existing_nodes = None

                        #node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

                        cur_adj = torch.sparse.LongTensor(cur_adj['idx'].T, cur_adj['vals'])
                        hist_adj_list.append(cur_adj)

                # This would be if we were training on all the edges in the time_window
                label_adj = tu.get_sp_adj(edges=self.data.edges,
                                        time=idx + 1,
                                        weighted=False,
                                        time_window=self.args.adj_mat_time_window)
                if test:
                    neg_mult = self.args.negative_mult_test
                else:
                    neg_mult = self.args.negative_mult_training

                if self.args.smart_neg_sampling:
                    existing_nodes = torch.cat(existing_nodes)

                if 'all_edges' in kwargs.keys() and kwargs['all_edges'] == True:
                    non_exisiting_adj = tu.get_all_non_existing_edges(adj=label_adj, tot_nodes=self.data.num_nodes)
                else:
                    non_exisiting_adj = tu.get_non_existing_edges(adj=label_adj,
                                                                number=label_adj['vals'].size(0) * neg_mult,
                                                                tot_nodes=self.data.num_nodes,
                                                                smart_sampling=self.args.smart_neg_sampling,
                                                                existing_nodes=existing_nodes)


                label_adj['idx'] = torch.cat([label_adj['idx'], non_exisiting_adj['idx']])
                label_adj['vals'] = torch.cat([label_adj['vals'], non_exisiting_adj['vals']])
                return {'idx': idx,
                        'edge_feature': hist_adj_list,
                        'label_sp': label_adj,
                        'node_feature': 1}
                
>>>>>>> 6b0fcad838f15c2e31807a46fde589b89552670e

    

