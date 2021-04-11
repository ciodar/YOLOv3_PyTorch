from darknet import Darknet
from utils import read_data_cfg
import numpy as np
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
        fm_dir = (pl.Path.joinpath(out_path, set))
        if not fm_dir.exists():
            fm_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving features into {str(fm_dir)}")
        eval_file = options[set]
        valid_dataset = dataset.densityDataset(eval_file, shape=(m.width, m.height),
                                               shuffle=False,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                               ]))

        kwargs = {'num_workers': 2, 'pin_memory': True}
        assert args.bs > 0
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.bs, shuffle=False, **kwargs)

        pbar = tqdm.tqdm(valid_loader)
        with torch.no_grad():
            for count_loop, (data, target, org_w, org_h) in enumerate(pbar):
                if use_cuda:
                    pbar.set_postfix({'GPU memory allocated': torch.cuda.memory_allocated(cuda_device) / (1024 * 1024)})
                    # print("%5d|GPU memory allocated: %.3f MB"%(count_loop,(torch.cuda.memory_allocated(cuda_device) / (1024 * 1024))))
                    data = data.to(cuda_device)

                output = m(data).numpy()
                np.save(str(pl.Path.joinpath(fm_dir,pl.Path(valid_dataset.get_image(count_loop)).stem)),output)
                with open(pl.Path.joinpath(fm_dir,pl.Path(valid_dataset.get_image(count_loop)).stem+'.txt'),'w') as f:
                    f.write(str(target.item()))

                del data, target, output
        # n_batches,batch,depth,height,width

if __name__ == '__main__':
    args = cmdline.arg_parse()
    # 1. get tensor
    feature_extraction(args)
