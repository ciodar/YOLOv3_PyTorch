import cmdline
from darknet import Darknet
from utils import read_data_cfg
import torch
from torchvision import transforms
import dataset
import tqdm
import pathlib as pl
from densitynet import DensityNet
import torch.optim as optim
import datetime
import torch.utils.data as D
import torch.nn as nn

def density(args):
    options = read_data_cfg(args.images)
    assert args.set in ['train', 'valid', 'test']
    eval_file = options[args.set]
    out_path = pl.Path(args.det)
    fm_file = (pl.Path.joinpath(out_path.parent, out_path.name))
    gt_file = (pl.Path.joinpath(out_path.parent, 'ground_truth_'+args.set+'_'+fm_file.name))
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
    kwargs = {'num_workers': 4, 'pin_memory': True}
    assert args.bs > 0
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.bs, shuffle=False, **kwargs)
    fm = []
    gt = []

    pbar = tqdm.tqdm(valid_loader)
    for count_loop, (data, target, org_w, org_h) in enumerate(pbar):
        if use_cuda:
            pbar.set_postfix({'GPU memory allocated': torch.cuda.memory_allocated(cuda_device) / (1024 * 1024)})
            #print("%5d|GPU memory allocated: %.3f MB"%(count_loop,(torch.cuda.memory_allocated(cuda_device) / (1024 * 1024))))
            data = data.cuda(cuda_device)
        # output = m(data).detach()
        # fm.append(output)
        gt.append(target.detach())

    # fm = torch.stack(fm)
    gt = torch.stack(gt)
    # n_batches,batch,depth,height,width
    # torch.save(fm, str(fm_file))
    torch.save(gt, str(gt_file))
def train(args):
    options = read_data_cfg(args.images)
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}.")
    train_path = pl.Path(options['train'])
    train_label_path = pl.Path.joinpath(train_path.parent,train_path.stem+'_labels'+train_path.suffix)
    t1 = torch.load(train_path, map_location=device).reshape(8862, 1792, 8, 10)
    t2 = torch.load(train_label_path, map_location=device).reshape(8862)
    trainset = list(zip(t1, t2))

    train_loader = D.DataLoader(trainset, batch_size=64,
                                shuffle=True)  # <1>

    model = DensityNet(512).to(device=device)  # <2>
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # <3>
    loss_fn = nn.MSELoss()  # <4>

    training_loop(  # <5>
        n_epochs=args.epoch,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        device=device
    )

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,device):

    for epoch in range(1, n_epochs + 1):  # <2>
        loss_train = 0.0
        for imgs, labels in train_loader:  # <3>
            imgs = imgs.to(device=device)
            labels = labels.to(device=device).view(-1,1).float()

            outputs = model(imgs)  # <4>

            loss = loss_fn(outputs, labels)  # <5>

            optimizer.zero_grad()  # <6>

            loss.backward()  # <7>

            optimizer.step()  # <8>

            loss_train += loss.item()  # <9>

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))  # <10>

def validate(model,args):
    options = read_data_cfg(args.images)
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Validating on device {device}.")
    train_path = str(pl.Path(options['train']))
    train_label_path = str(pl.Path.joinpath(train_path.parent(), train_path.stem() + '_labels' + train_path.ext()))
    t1 = torch.load(train_path, map_location=device).reshape(8862, 1792, 8, 10)
    t2 = torch.load(train_label_path, map_location=device).reshape(8862)
    trainset = list(zip(t1, t2))

    train_loader = D.DataLoader(trainset, batch_size=64,
                                shuffle=False)  # <1>

    valid_path = str(pl.Path(options['valid']))
    valid_label_path = str(pl.Path.joinpath(train_path.parent(), train_path.stem() + '_labels' + train_path.ext()))

    v1 = torch.load(valid_path, map_location=device).reshape(8862, 1792, 8, 10)
    v2 = torch.load(valid_label_path, map_location=device).reshape(8862)
    valset = list(zip(v1, v2))
    val_loader = torch.utils.data.DataLoader(valset, batch_size=64,
                                             shuffle=False)

    for name, loader in [("train", train_loader), ("val", val_loader)]:
        mse = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device).float()
                outputs = model(imgs)
                total += labels.shape[0]
                mse += ((outputs-labels)**2).sum()
        print("Accuracy {}: {:.2f}".format(name , mse / total))

if __name__ == '__main__':
    args = cmdline.arg_parse()
    #1. get tensor
    # density(args)
    #2. evaluation
    # tensor = torch.load(tensorpath,map_location=torch.device("cpu")).reshape(8862,1792,8,10)
    # print(tensor)
    # m = DensityNet()
    # m.eval()
    # output = m(tensor[0].unsqueeze(0))
    # print(output)
    #3. training
    train(args)







