import cmdline
from darknet import Darknet
from utils import image2torch
from utils import get_all_boxes, bbox_iou, nms, read_data_cfg, load_class_names, get_image_list
import cv2
import torch
from torchvision import datasets, transforms
import dataset
import tqdm

def density(args):
    options = read_data_cfg(args.datacfg)
    train_file = options['train']
    m = Darknet(args.cfgfile)
    m.print_network()
    check_model = args.cfgfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(args.cfgfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(args.weightsfile)

    img = cv2.imread(args.images)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    use_cuda = torch.cuda.is_available() and (True if args.cuda is None else args.cuda)
    cuda_device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        m.cuda(cuda_device)
        #print("Using device #", device_n, " (", get_device_name(cuda_device), ")")
    m.eval()

    valid_dataset = dataset.listDataset(train_file, shape=(m.width, m.height),
                                        shuffle=False,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                        ]))
    kwargs = {'num_workers': 4, 'pin_memory': True}
    valid_batchsize = 2
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=valid_batchsize, shuffle=False, **kwargs)
    output = []

    for count_loop, (data, target, org_w, org_h) in enumerate(tqdm.tqdm(valid_loader)):
        if use_cuda:
            data = data.cuda(cuda_device)
        output.append(m(data))
    output = torch.stack(output)
    # n_batches,batch,depth,height,width
    torch.save(output,args.det)


if __name__ == '__main__':
    args = cmdline.arg_parse()
    density(args)
    tensor = torch.load(args.det)
    print(tensor)