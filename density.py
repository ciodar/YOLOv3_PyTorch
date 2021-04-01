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
import matplotlib.pyplot as plt

train_losses=[]
train_accu=[]
eval_losses=[]
eval_accu=[]

def feature_extraction(args):
    options = read_data_cfg(args.images)
    assert args.set in ['train', 'valid', 'test']
    eval_file = options[args.set]
    out_path = pl.Path(args.det)
    fm_file = (pl.Path.joinpath(out_path.parent, out_path.name))
    gt_file = (pl.Path.joinpath(out_path.parent, fm_file.stem + '_' + args.set + '_labels' + fm_file.suffix))
    if not fm_file.parent.exists() or not gt_file.parent.exists():
        raise Exception("Selected output path does not exist")

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
    fm = torch.empty(2, 1792, 16, 20).to(cuda_device)
    gt = torch.empty(2).to(cuda_device)
    if use_cuda:
        m.to(cuda_device)
        # print("Using device #", cuda_device, " (", get_device_name(cuda_device), ")")
    m.eval()

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
                data = data.cuda(cuda_device)
            fm = torch.cat((fm, m(data).clone().detach().to(cuda_device)))
            gt = torch.cat((gt, target.clone().detach().to(cuda_device)))
            del data, target
    # n_batches,batch,depth,height,width
    torch.save(fm, str(fm_file))
    torch.save(gt, str(gt_file))

def density(args):
    model = DensityNet()
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Training on device {device}, lr={args.lr:e}.")
    options = read_data_cfg(args.images)
    train_path = pl.Path(options['train'])
    valid_path = pl.Path(options['valid'])

    # train loader
    train_label_path = pl.Path.joinpath(train_path.parent, train_path.stem + '_labels' + train_path.suffix)
    t1 = torch.load(train_path, map_location=device).reshape(8862, 1792, 8, 10)
    t2 = torch.load(train_label_path, map_location=device).reshape(8862)
    trainset = list(zip(t1, t2))
    train_loader = D.DataLoader(trainset, batch_size=64,
                                shuffle=True)

    # test loader
    valid_label_path = pl.Path.joinpath(valid_path.parent, valid_path.stem + '_labels' + valid_path.suffix)
    v1 = torch.load(valid_path, map_location=device).reshape(1366, 1792, 8, 10)
    v2 = torch.load(valid_label_path, map_location=device).reshape(1366)
    valset = list(zip(v1, v2))
    valid_loader = torch.utils.data.DataLoader(valset, batch_size=64,
                                             shuffle=False)

    # retrieve weights
    # model.load_state_dict(torch.load('D:/results/10_18_2conv_norm.pt',map_location="cpu"))
    train(model, args,train_loader,valid_loader,device)
    #if args.save:
    #    torch.save(model.state_dict(), pl.Path.joinpath(train_path.parent, 'trained_model.pt'))

    fig = plt.figure()
    plt.ylim(0,10)
    plt.plot(train_losses, '-o')
    plt.plot(eval_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.ylim(0, 10)
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig(pl.Path.joinpath(pl.Path.joinpath(train_path.parent,'train_loss.png')))

def train(model,args,train_loader,valid_loader,device):

    model.to(device=device)  # <2>
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # <3>
    loss_fn = nn.MSELoss()  # <4>

    training_loop(  # <5>
        n_epochs=args.epoch,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader = valid_loader,
        device=device
    )

    return model


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,valid_loader, device):
    for epoch in range(1, n_epochs + 1):  # <2>
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for data, labels in train_loader:  # <3>
            data = data.to(device=device)
            labels = labels.to(device=device).view(-1, 1).float()

            outputs = model(data)  # <4>

            loss = loss_fn(outputs, labels)  # <5>

            optimizer.zero_grad()  # <6>

            loss.backward()  # <7>

            optimizer.step()  # <8>

            running_loss += loss.item()  # <9>
            total += labels.size(0)
            correct += outputs.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        accu = 100. * correct / total

        train_accu.append(accu)
        train_losses.append(train_loss)
        test(model,valid_loader,device,loss_fn)

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {} GPU usage {} '.format(
                datetime.datetime.now(), epoch,
                train_loss,
                torch.cuda.max_memory_allocated(device) / 1024 ** 2))  # <10>






def validate(model, args, train_loader, valid_loader):
    options = read_data_cfg(args.images)
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))
    print(f"Validating on device {device}.")

    model.eval()

    for name, loader in [("train", train_loader), ("val", valid_loader)]:
        mse = 0.
        total = 0.
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(device=device)
                labels = labels.to(device=device).float()
                outputs = model(data).view(-1)
                total += labels.shape[0]
                mse += ((outputs - labels) ** 2).sum()
        print("MSE {}: {:.2f}".format(name, mse / total))


def test(model,valid_loader,device,loss_fn):
    model.eval()

    running_loss = 0.
    correct = 0.
    total = 0.

    with torch.no_grad():
        for data,labels in valid_loader:
            data = data.to(device=device)
            labels = labels.to(device=device).view(-1, 1).float()
            outputs = model(data)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    valid_loss = running_loss / len(valid_loader)
    accu = 100. * correct / total

    eval_losses.append(valid_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f' % (valid_loss, accu))


if __name__ == '__main__':
    args = cmdline.arg_parse()
    # 1. get tensor
    # feature_extraction(args)
    # 2. evaluation
    # tensor = torch.load(tensorpath,map_location=torch.device("cpu")).reshape(8862,1792,8,10)
    # print(tensor)
    # m = DensityNet()
    # m.eval()
    # output = m(tensor[0].unsqueeze(0))
    # print(output)
    # 3. training
    density(args)
    #validate(model,args)