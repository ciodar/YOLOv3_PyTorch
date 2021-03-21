import cmdline
from darknet import Darknet
from utils import read_data_cfg,convert2cpu
import torch
from torchvision import transforms
import dataset
import tqdm
import pathlib as pl

def density(args):
    options = read_data_cfg(args.images)
    assert args.set in ['train', 'valid', 'test']
    eval_file = options[args.set]
    out_path = pl.Path(args.det)
    fm_file = (pl.Path.joinpath(out_path.parent, out_path.name))
    gt_file = (pl.Path.joinpath(out_path.parent, 'ground_truth_'+fm_file.name))
    if not fm_file.parent.exists() or not gt_file.parent.exists():
        raise Exception("Selected output path does not exist")

    m = Darknet(args.cfgfile)
    #m.print_network()
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
        m.cuda(cuda_device)
        #print("Using device #", cuda_device, " (", get_device_name(cuda_device), ")")
    m.eval()

    valid_dataset = dataset.densityDataset(eval_file, shape=(m.width, m.height),
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    kwargs = {'num_workers': 2, 'pin_memory': False}
    assert args.bs > 0
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.bs, shuffle=False, **kwargs)
    fm = []
    gt = []

    for count_loop, (data, target, org_w, org_h) in enumerate(tqdm.tqdm(valid_loader)):
        if use_cuda:
            print("%5d|GPU memory allocated: %.3f MB"%(count_loop,(torch.cuda.memory_allocated(cuda_device) / (1024 * 1024))))
            data = data.cuda(cuda_device)
        output = convert2cpu(m(data))
        fm.append(output)
        gt.append(target)

    output = torch.stack(output)
    gt = torch.stack(gt)
    # n_batches,batch,depth,height,width
    torch.save(output, args.det)
    torch.save(output, str(gt_file))


if __name__ == '__main__':
    args = cmdline.arg_parse()
    density(args)
    tensor = torch.load(args.det)
    print(tensor)
