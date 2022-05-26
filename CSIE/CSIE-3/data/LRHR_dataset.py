import torch.utils.data as data

from data import common


class LRHRDataset(data.Dataset):
    '''
    Read LR and HR images in train and eval phases.
    '''

    def name(self):
        return common.find_benchmark(self.opt['dataroot_LR1'])


    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'
        self.scale = self.opt['scale']
        self.paths_HR, self.paths_LR1, self.paths_LR2 = None, None, None

        # change the length of train dataset (influence the number of iterations in each epoch)
        self.repeat = 2

        # read image list from image/binary files
        self.paths_HR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_HR'])
        self.paths_LR1 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR1'])
        self.paths_LR2 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR2'])
        self.paths_LR3 = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_LR2'])

        assert self.paths_HR, '[Error] HR paths are empty.'
        if self.paths_LR1 and self.paths_LR2 and self.paths_LR3 and self.paths_HR:
            assert len(self.paths_LR1) == len(self.paths_HR) and len(self.paths_LR2) == len(self.paths_HR) and len(self.paths_LR3) == len(self.paths_HR), \
                '[Error] HR: [%d], LR1: [%d], LR2: [%d], and LR3: [%d] have different number of images.'%(
                len(self.paths_HR), len(self.paths_LR1), len(self.paths_LR2), len(self.paths_LR3))


    def __getitem__(self, idx):
        lr1,lr2, lr3, hr, lr_path1, lr_path2, lr_path3, hr_path = self._load_file(idx)
        if self.train:
            lr1, lr2, lr3, hr = self._get_patch(lr1,lr2, lr3, hr)
        lr_tensor1, lr_tensor2,lr_tensor3, hr_tensor = common.np2Tensor([lr1,lr2, lr3, hr], self.opt['rgb_range'])
        return {'LR1': lr_tensor1,'LR2': lr_tensor2, 'LR3': lr_tensor3, 'HR': hr_tensor, 'LR_path1': lr_path1, 'LR_path2': lr_path2, 'LR_path3': lr_path3, 'HR_path': hr_path}


    def __len__(self):
        if self.train:
            return len(self.paths_HR) * self.repeat
        else:
            return len(self.paths_LR1)


    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_HR)
        else:
            return idx


    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr_path1 = self.paths_LR1[idx]
        lr_path2 = self.paths_LR2[idx]
        lr_path3 = self.paths_LR3[idx]
        hr_path = self.paths_HR[idx]
        lr1 = common.read_img_fromtxt(lr_path1, self.opt['data_type'])
        lr2 = common.read_img_fromtxt(lr_path2, self.opt['data_type'])
        lr3 = common.read_img_fromtxt(lr_path3, self.opt['data_type'])
        hr = common.read_img_fromtxt(hr_path, self.opt['data_type'])

        return lr1,lr2, lr3, hr, lr_path1, lr_path2, lr_path3, hr_path


    def _get_patch(self, lr1, lr2, lr3, hr):

        LR_size = self.opt['LR_size']
        # random crop and augment
        lr1, lr2, lr3, hr = common.get_patch(lr1, lr2, lr3, hr, LR_size, self.scale)
        lr1, lr2, lr3, hr = common.augment([lr1, lr2, lr3, hr])

        return lr1, lr2, lr3, hr
