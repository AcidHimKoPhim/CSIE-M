import argparse, time, os
import imageio

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))
    f = open(opt['savefile'], "w")
    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]"%bm)

        sr_list = []
        rec_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_psnr_input = []
        total_ssim_input = []
        total_time = []

        need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)
            
            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)
            #visuals_rec = solver.get_current_visual(need_HR=need_HR)
            sr_list.append(visuals['SR'])
            
            
            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                path_list.append(os.path.basename(batch['HR_path'][0]).replace('HR', model_name))
            else:
                path_list.append(os.path.basename(batch['LR_path1'][0]))
            #m = os.path.basename(batch['LR_path1'][0])+"\t"+str((t1 - t0))+"\n"
                
        
        # save SR results for further evaluation on MATLAB
        
        save_img_path = opt['dir']

        print("===> Saving SR images of [%s]... Save Path: [%s]\n" % (bm, save_img_path))
        
        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            f.write(os.path.join(save_img_path, name)+'\n')
            imageio.imwrite(os.path.join(save_img_path, name), img)

    print("==================================================")
    print("===> Finished !")

if __name__ == '__main__':
    main()