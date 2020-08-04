from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class splitter:
    def __init__(self, args, tasker, sample='False'):
        assert args.train_proportion + args.dev_proportion < 1, \
            'there\'s no space for test samples'
        # only the training one requires special handling on start, the others are fine with the split IDX.
        start = tasker.data.min_time + args.num_hist_steps - 1  # -1 + args.adj_mat_time_window
        end = args.train_proportion

        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(tasker, start, end, test=False)
        train = DataLoader(train, **args.data_loading_params)

        start = end
        end = args.dev_proportion + args.train_proportion
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        if args.task == 'link_pred':
            dev = data_split(tasker, start, end, test=True)
        else:
            dev = data_split(tasker, start, end, test=True)
        #print(dev)
        dev = DataLoader(dev, num_workers=args.data_loading_params['num_workers'])

        start = end

        # the +1 is because I assume that max_time exists in the dataset
        end = int(tasker.max_time) + 1
        if args.task == 'link_pred':
            test = data_split(tasker, start, end, test=True)
        else:
            test = data_split(tasker, start, end, test=True)

        test = DataLoader(test, num_workers=args.data_loading_params['num_workers'])

        print('Dataset splits sizes:  train', len(train), 'dev', len(dev), 'test', len(test))

        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test



class data_split(Dataset):
    def __init__(self, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, test=self.test, **self.kwargs)
        #print(t)
        return t
