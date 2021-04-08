from darknet import Darknet
from utils import read_data_cfg
import pathlib as pl
import torch
import dataset
import tqdm
import cmdline
from torchvision import transforms

def feature_extraction(args):
    options = read_data_cfg(args.images)
    assert args.set in ['train', 'valid', 'test']
    eval_file = options[args.set]
    out_path = pl.Path(args.det)
    fm_file = (pl.Path.joinpath(out_path.parent, out_path.name))
    gt_file = (pl.Path.joinpath(out_path.parent, fm_file.stem + '_' + args.set + '_labels' + fm_file.suffix))
    if not fm_file.parent.exists() or not gt_file.parent.exists():
        raise Exception("Selected output path does not exist")
    print("Saving features into %s,labels into %s" % (str(fm_file), str(gt_file)))
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

    valid_dataset = dataset.densityDataset(eval_file, shape=(m.width, m.height),
                                           shuffle=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                           ]))

    fm = []
    gt = []
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
                data = data.cuda(cuda_device)
            output = m(data).clone().detach()
            fm.append(output)
            gt.append(target.clone().detach())
            del data, target, output
    # n_batches,batch,depth,height,width
    fm = torch.stack(fm).reshape(len(valid_dataset), 1792, 8, 10)
    gt = gt.stack(gt).reshape(len(valid_dataset))

    torch.save(fm, str(fm_file))
    torch.save(gt, str(gt_file))


if __name__ == '__main__':
    args = cmdline.arg_parse()
    # 1. get tensor
    feature_extraction(args)
