import utils as u
import os

import tarfile

import torch
from datetime import datetime


class UCI:
    def __init__(self, args):
        args.uc_irc_args = u.Namespace(args.uc_irc_args)

        tar_file = os.path.join(args.uc_irc_args.folder, args.uc_irc_args.tar_file)
        tar_archive = tarfile.open(tar_file, 'r:bz2')

        self.edges = self.load_edges(args, tar_archive)

    def load_edges(self, args, tar_archive):
        data = u.load_data_from_tar(args.uc_irc_args.edges_file,
                                    tar_archive,
                                    starting_line=2,
                                    sep=' ')
        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'weight': 2,
                            'time': 3})

        data = data.long()

        self.num_nodes = int(data[:, [cols.source, cols.target]].max())

        # first id should be 0 (they are already contiguous)
        data[:, [cols.source, cols.target]] -= 1

        # add edges in the other direction (simmetric)
        data = torch.cat([data,
                          data[:, [cols.target,
                                   cols.source,
                                   cols.weight,
                                   cols.time]]],
                         dim=0)

        data[:, cols.time] = u.aggregate_by_time(data[:, cols.time],
                                                 args.uc_irc_args.aggr_time)

        ids = data[:, cols.source] * self.num_nodes + data[:, cols.target]
        self.num_non_existing = float(self.num_nodes ** 2 - ids.unique().size(0))

        idx = data[:, [cols.source,
                       cols.target,
                       cols.time]]

        self.max_time = data[:, cols.time].max()
        self.min_time = data[:, cols.time].min()

        return {'idx': idx, 'vals': torch.ones(idx.size(0))}


class sbm:
    def __init__(self, args):
        assert args.task in ['link_pred'], 'sbm only implements link_pred'
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                  })
        args.sbm_args = u.Namespace(args.sbm_args)

        # build edge data structure
        edges = self.load_edges(args.sbm_args)
        timesteps = u.aggregate_by_time(edges[:, self.ecols.TimeStep], args.sbm_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        print ('TIME', self.max_time, self.min_time)
        edges[:, self.ecols.TimeStep] = timesteps

        edges[:, self.ecols.Weight] = self.cluster_negs_and_positives(edges[:, self.ecols.Weight])
        self.num_classes = edges[:, self.ecols.Weight].unique().size(0)

        self.edges = self.edges_to_sp_dict(edges)

        # random node features
        self.num_nodes = int(self.get_num_nodes(edges))
        #self.feats_per_node = args.sbm_args.feats_per_node
        #self.nodes_feats = torch.rand((self.num_nodes, self.feats_per_node))

        self.num_non_existing = self.num_nodes ** 2 - edges.size(0)

    def cluster_negs_and_positives(self, ratings):
        pos_indices = ratings >= 0
        neg_indices = ratings < 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings

    def prepare_node_feats(self, node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self, edges):
        idx = edges[:, [self.ecols.FromNodeId,
                        self.ecols.ToNodeId,
                        self.ecols.TimeStep]]

        vals = edges[:, self.ecols.Weight]

        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self, edges):
        all_ids = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self, sbm_args, starting_line=1):
        file = os.path.join(sbm_args.folder, sbm_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
        edges = torch.tensor(edges, dtype=torch.long)
        return edges

    def make_contigous_node_ids(self, edges):
        new_edges = edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:, [self.ecols.FromNodeId, self.ecols.ToNodeId]] = new_edges
        return edges


class Reddit:
    def __init__(self, args):
        args.reddit_args = u.Namespace(args.reddit_args)
        folder = args.reddit_args.folder

        # load nodes
        cols = u.Namespace({'id': 0,
                            'feats': 1})
        file = args.reddit_args.nodes_file
        file = os.path.join(folder, file)
        with open(file) as file:
            file = file.read().splitlines()

        ids_str_to_int = {}
        id_counter = 0

        feats = []

        for line in file:
            line = line.split(',')
            # node id
            nd_id = line[0]
            if nd_id not in ids_str_to_int.keys():
                ids_str_to_int[nd_id] = id_counter
                id_counter += 1
                nd_feats = [float(r) for r in line[1:]]
                feats.append(nd_feats)
            else:
                print('duplicate id', nd_id)
                raise Exception('duplicate_id')

        feats = torch.tensor(feats, dtype=torch.float)
        num_nodes = feats.size(0)

        edges = []
        not_found = 0

        # load edges in title
        edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.title_edges_file,
                                                             folder,
                                                             ids_str_to_int)
        edges.extend(edges_tmp)
        not_found += not_found_tmp

        # load edges in bodies

        edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.body_edges_file,
                                                             folder,
                                                             ids_str_to_int)
        edges.extend(edges_tmp)
        not_found += not_found_tmp

        # min time should be 0 and time aggregation
        edges = torch.LongTensor(edges)
        edges[:, 2] = u.aggregate_by_time(edges[:, 2], args.reddit_args.aggr_time)
        max_time = edges[:, 2].max()

        # separate classes
        sp_indices = edges[:, :3].t()
        sp_values = edges[:, 3]

        # sp_edges = torch.sparse.LongTensor(sp_indices
        # 									  ,sp_values,
        # 									  torch.Size([num_nodes,
        # 									  			  num_nodes,
        # 									  			  max_time+1])).coalesce()
        # vals = sp_edges._values()
        # print(vals[vals>0].sum() + vals[vals<0].sum()*-1)
        # asdf

        pos_mask = sp_values == 1
        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:, neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                               , neg_sp_values,
                                               torch.Size([num_nodes,
                                                           num_nodes,
                                                           max_time + 1])).coalesce()

        pos_sp_indices = sp_indices[:, pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                               , pos_sp_values,
                                               torch.Size([num_nodes,
                                                           num_nodes,
                                                           max_time + 1])).coalesce()

        # scale positive class to separate after adding
        pos_sp_edges *= 1000

        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        # separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals % 1000
        pos_vals = vals // 1000
        # vals is simply the number of edges between two nodes at the same time_step, regardless of the edge label
        vals = pos_vals - neg_vals

        # creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0), dtype=torch.long)
        new_vals[vals > 0] = 1
        new_vals[vals <= 0] = 0
        vals = pos_vals + neg_vals
        indices_labels = torch.cat([sp_edges._indices().t(), new_vals.view(-1, 1)], dim=1)

        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_classes = 2
        self.feats_per_node = feats.size(1)
        self.num_nodes = num_nodes
        self.nodes_feats = feats
        self.max_time = max_time
        self.min_time = 0

    def prepare_node_feats(self, node_feats):
        node_feats = node_feats[0]
        return node_feats

    def load_edges_from_file(self, edges_file, folder, ids_str_to_int):
        edges = []
        not_found = 0

        file = edges_file

        file = os.path.join(folder, file)
        with open(file) as file:
            file = file.read().splitlines()

        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'time': 3,
                            'label': 4})

        base_time = datetime.strptime("19800101", '%Y%m%d')

        for line in file[1:]:
            fields = line.split('\t')
            sr = fields[cols.source]
            tg = fields[cols.target]

            if sr in ids_str_to_int.keys() and tg in ids_str_to_int.keys():
                sr = ids_str_to_int[sr]
                tg = ids_str_to_int[tg]

                time = fields[cols.time].split(' ')[0]
                time = datetime.strptime(time, '%Y-%m-%d')
                time = (time - base_time).days

                label = int(fields[cols.label])
                edges.append([sr, tg, time, label])
                # add the other edge to make it undirected
                edges.append([tg, sr, time, label])
            else:
                not_found += 1

        return edges, not_found


class Elliptic:
    def __init__(self,args):
        args.elliptic_args = u.Namespace(args.elliptic_args)

        tar_file = os.path.join(args.elliptic_args.folder, args.elliptic_args.tar_file)
        tar_archive = tarfile.open(tar_file, 'r:gz')

        self.nodes_labels_times = self.load_node_labels(args.elliptic_args, tar_archive)

        self.edges = self.load_transactions(args.elliptic_args, tar_archive)

        self.nodes, self.nodes_feats = self.load_node_feats(args.elliptic_args, tar_archive)

    def load_node_feats(self, elliptic_args, tar_archive):
        data = u.load_data_from_tar(elliptic_args.feats_file, tar_archive, starting_line=0)
        nodes = data

        nodes_feats = nodes[:,1:]


        self.num_nodes = len(nodes)
        self.feats_per_node = data.size(1) - 1

        return nodes, nodes_feats.float()


    def load_node_labels(self, elliptic_args, tar_archive):
        labels = u.load_data_from_tar(elliptic_args.classes_file, tar_archive, replace_unknow=True).long()
        times = u.load_data_from_tar(elliptic_args.times_file, tar_archive, replace_unknow=True).long()
        lcols = u.Namespace({'nid': 0,
                             'label': 1})
        tcols = u.Namespace({'nid':0, 'time':1})


        nodes_labels_times =[]
        for i in range(len(labels)):
            label = labels[i,[lcols.label]].long()
            if label>=0:
                nid=labels[i,[lcols.nid]].long()
                time=times[nid,[tcols.time]].long()
                nodes_labels_times.append([nid , label, time])
        nodes_labels_times = torch.tensor(nodes_labels_times)

        return nodes_labels_times


    def load_transactions(self, elliptic_args, tar_archive):
        data = u.load_data_from_tar(elliptic_args.edges_file, tar_archive, type_fn=float, tensor_const=torch.LongTensor)
        tcols = u.Namespace({'source': 0,
                             'target': 1,
                             'time': 2})

        data = torch.cat([data,data[:,[1,0,2]]])

        self.max_time = data[:,tcols.time].max()
        self.min_time = data[:,tcols.time].min()

        return {'idx': data, 'vals': torch.ones(data.size(0))}


class Bitcoin:
    def __init__(self,args):
        assert args.task in ['link_pred', 'edge_cls'], 'bitcoin only implements link_pred or edge_cls'
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.bitcoin_args = u.Namespace(args.bitcoin_args)

        #build edge data structure
        edges = self.load_edges(args.bitcoin_args)

        edges = self.make_contigous_node_ids(edges)
        num_nodes = edges[:,[self.ecols.FromNodeId,
                            self.ecols.ToNodeId]].unique().size(0)

        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep],args.bitcoin_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])


        #add the reversed link to make the graph undirected
        edges = torch.cat([edges,edges[:,[self.ecols.ToNodeId,
                                          self.ecols.FromNodeId,
                                          self.ecols.Weight,
                                          self.ecols.TimeStep]]])

        #separate classes
        sp_indices = edges[:,[self.ecols.FromNodeId,
                              self.ecols.ToNodeId,
                              self.ecols.TimeStep]].t()
        sp_values = edges[:,self.ecols.Weight]


        neg_mask = sp_values == -1

        neg_sp_indices = sp_indices[:,neg_mask]
        neg_sp_values = sp_values[neg_mask]
        neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
                                              ,neg_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        pos_mask = sp_values == 1

        pos_sp_indices = sp_indices[:,pos_mask]
        pos_sp_values = sp_values[pos_mask]

        pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
                                              ,pos_sp_values,
                                              torch.Size([num_nodes,
                                                          num_nodes,
                                                          self.max_time+1])).coalesce()

        #scale positive class to separate after adding
        pos_sp_edges *= 1000

        #we substract the neg_sp_edges to make the values positive
        sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()

        #separating negs and positive edges per edge/timestamp
        vals = sp_edges._values()
        neg_vals = vals%1000
        pos_vals = vals//1000
        #We add the negative and positive scores and do majority voting
        vals = pos_vals - neg_vals
        #creating labels new_vals -> the label of the edges
        new_vals = torch.zeros(vals.size(0),dtype=torch.long)
        new_vals[vals>0] = 1
        new_vals[vals<=0] = 0
        indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)

        #the weight of the edges (vals), is simply the number of edges between two entities at each time_step
        vals = pos_vals + neg_vals


        self.edges = {'idx': indices_labels, 'vals': vals}
        self.num_nodes = num_nodes
        self.num_classes = 2


    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings > 0
        neg_indices = ratings <= 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = -1
        return ratings

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        idx = edges[:,[self.ecols.FromNodeId,
                       self.ecols.ToNodeId,
                       self.ecols.TimeStep]]

        vals = edges[:,self.ecols.Weight]
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self,bitcoin_args):
        file = os.path.join(bitcoin_args.folder,bitcoin_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges


class AS:
    def __init__(self, args):
        args.aut_sys_args = u.Namespace(args.aut_sys_args)

        tar_file = os.path.join(args.aut_sys_args.folder, args.aut_sys_args.tar_file)
        tar_archive = tarfile.open(tar_file, 'r:gz')

        self.edges = self.load_edges(args, tar_archive)

    def load_edges(self, args, tar_archive):
        files = tar_archive.getnames()

        cont_files2times = self.times_from_names(files)

        edges = []
        cols = u.Namespace({'source': 0,
                            'target': 1,
                            'time': 2})
        for file in files:
            data = u.load_data_from_tar(file,
                                        tar_archive,
                                        starting_line=4,
                                        sep='\t',
                                        type_fn=int,
                                        tensor_const=torch.LongTensor)

            time_col = torch.zeros(data.size(0), 1, dtype=torch.long) + cont_files2times[file]

            data = torch.cat([data, time_col], dim=1)

            data = torch.cat([data, data[:, [cols.target,
                                             cols.source,
                                             cols.time]]])

            edges.append(data)

        edges = torch.cat(edges)

        _, edges[:, [cols.source, cols.target]] = edges[:, [cols.source, cols.target]].unique(return_inverse=True)

        # use only first X time steps
        indices = edges[:, cols.time] < args.aut_sys_args.steps_accounted
        edges = edges[indices, :]

        # time aggregation
        edges[:, cols.time] = u.aggregate_by_time(edges[:, cols.time], args.aut_sys_args.aggr_time)

        self.num_nodes = int(edges[:, [cols.source, cols.target]].max() + 1)

        ids = edges[:, cols.source] * self.num_nodes + edges[:, cols.target]
        self.num_non_existing = float(self.num_nodes ** 2 - ids.unique().size(0))

        self.max_time = edges[:, cols.time].max()
        self.min_time = edges[:, cols.time].min()

        return {'idx': edges, 'vals': torch.ones(edges.size(0))}

    def times_from_names(self, files):
        files2times = {}
        times2files = {}

        base = datetime.strptime("19800101", '%Y%m%d')
        for file in files:
            delta = (datetime.strptime(file[2:-4], '%Y%m%d') - base).days

            files2times[file] = delta
            times2files[delta] = file

        cont_files2times = {}

        sorted_times = sorted(files2times.values())
        new_t = 0

        for t in sorted_times:
            file = times2files[t]

            cont_files2times[file] = new_t

            new_t += 1
        return cont_files2times