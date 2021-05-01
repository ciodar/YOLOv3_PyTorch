from darknet import Darknet
from utils import read_data_cfg
import pathlib as pl
import torch
import dataset
import tqdm
import cmdline
from torchvision import transforms

sets = ['train','valid']

def feature_extraction(args):
    options = read_data_cfg(args.images)
    out_path = pl.Path(args.det)
    nclasses = int(options['classes'])

    m = Darknet(args.cfgfile)
    # m.print_network()
    check_model = args.cfgfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(args.cfgfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(args.weightsfile)
    use_cuda = torch.cuda.is_available() and (True if args.cuda is None else args.cuda)
    cuda_device = torch.device(args.device if use_cuda else "cpu")

    if use_cuda:
        m.to(cuda_device)
        # print("Using device #", cuda_device, " (", get_device_name(cuda_device), ")")
    m.eval()
    for set in sets:
        fm_file = (pl.Path.joinpath(out_path,set+'_features.pt'))
        gt_file = (pl.Path.joinpath(out_path if out_path.is_dir() else out_path.parent, fm_file.stem + '_' + 'labels' + fm_file.suffix))
        if not fm_file.parent.exists():
            fm_file.mkdir(parents=True, exist_ok=True)
        print(f"Saving features into {str(fm_file)}")
        eval_file = options[set]
        valid_dataset = dataset.densityDataset(eval_file, shape=(m.width, m.height),
                                               shuffle=False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                               ]))
        batch_size = args.bs
        assert batch_size > 0
        fm = torch.zeros(len(valid_dataset)//batch_size + 1,batch_size,1792,8,10).to(device=cuda_device)
        gt = torch.zeros(len(valid_dataset)//batch_size + 1,batch_size, nclasses).to(device=cuda_device)
        kwargs = {'num_workers': 2, 'pin_memory': True}

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.bs, shuffle=False, **kwargs)

        pbar = tqdm.tqdm(valid_loader)
        with torch.no_grad():
            for count_loop, (data, target, org_w, org_h) in enumerate(pbar):
                if use_cuda:
                    pbar.set_postfix({'GPU memory allocated': torch.cuda.memory_allocated(cuda_device) / (1024 * 1024)})
                    # print("%5d|GPU memory allocated: %.3f MB"%(count_loop,(torch.cuda.memory_allocated(cuda_device) / (1024 * 1024))))
                    data = data.to(cuda_device)
                output = m(data).clone().detach()
                fm[count_loop,:,:,:,:]=output
                gt[count_loop,:] = target.clone().detach()[:,0:nclasses]
                del data, target, output
        # n_batches,batch,depth,height,width
        fm = fm[0:len(valid_dataset), :, :, :].reshape(len(valid_dataset), 1792, 8, 10)
        gt = gt[0:len(valid_dataset),:].reshape(len(valid_dataset),nclasses)

        torch.save(fm, fm_file)
        print(f"Saved feature maps into {str(fm_file)},shape:{fm.shape}")
        torch.save(gt, gt_file)
        print(f"Saved feature maps into {str(gt_file)},shape:{gt.shape}")
        del gt,fm



if __name__ == '__main__':
    args = cmdline.arg_parse()
    # 1. get tensor
    feature_extraction(args)