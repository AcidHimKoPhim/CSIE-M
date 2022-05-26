import torch.utils.data as data

from data import common


class LRfromtxt(data.Dataset):
    '''
    Read LR images only in test phase.
    '''

    def name(self):
        return common.find_benchmark(self.opt['LRtxtpath'])


    def __init__(self, opt):
        super(LRfromtxt, self).__init__()
        self.opt = opt
        self.scale = self.opt['scale']
        self.paths_LR1 = None
        self.paths_LR2 = None
        self.paths_LR3 = None
        self.list_LR1 = []
        self.list_LR2 = []
        self.list_LR3 = []
        
        # read image list from image/binary files
        self.paths_LR1, self.paths_LR2, self.paths_LR3 = common.get_image_paths_from_txt(self.opt['LRtxtpath'],1)

        for i in range(len(self.paths_LR1)):
            lr_path1 = self.paths_LR1[i]
            lr_path2 = self.paths_LR2[i]
            lr_path3 = self.paths_LR3[i]
            lr1 = common.read_img_fromtxt(lr_path1, self.opt['data_type'])
            lr2 = common.read_img_fromtxt(lr_path2, self.opt['data_type'])
            lr3 = common.read_img_fromtxt(lr_path3, self.opt['data_type'])
            self.list_LR1.append(lr1) 
            self.list_LR2.append(lr2) 
            self.list_LR3.append(lr3) 
        
        assert self.paths_LR1, '[Error] LR paths are empty.'


    def __getitem__(self, idx):
        # get LR image
        lr1=  self.list_LR1[idx]
        lr2=  self.list_LR2[idx]
        lr3=  self.list_LR3[idx]
        lr_path1 = self.paths_LR1[idx]
        lr_path2 = self.paths_LR2[idx]
        lr_path3 = self.paths_LR3[idx]
        lr_tensor1, lr_tensor2, lr_tensor3 = common.np2Tensor([lr1,lr2,lr3], self.opt['rgb_range'])
        return {'LR1': lr_tensor1, 'LR_path1': lr_path1, 'LR2': lr_tensor2, 'LR_path2': lr_path2, 'LR3': lr_tensor3, 'LR_path3': lr_path3}


    def __len__(self):
        return len(self.paths_LR1)