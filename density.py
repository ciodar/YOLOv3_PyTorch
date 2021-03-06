import cmdline
from datetime import datetime
from utils import read_data_cfg
import torch
import tqdm
import pathlib as pl
from densitynet import DensityNet
import torch.optim as optim
import torch.utils.data as D
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.metrics import mean_squared_error

train_losses = []
train_accu = []
eval_losses = []
eval_accu = []

def train(model, args, train_loader, valid_loader, device):
    model.to(device=device)  # <2>
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # <3>
    loss_fn = nn.MSELoss()  # <4>

    training_loop(  # <5>
        n_epochs=args.epoch,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device
    )


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, valid_loader, device):
    pbar = tqdm.tqdm(range(1, n_epochs + 1))
    for epoch in pbar:  # <2>

        model.train()

        running_loss = 0.0
        correct = 0.0
        total = 0.0
        for data, labels in train_loader:  # <3>
            data = data.to(device=device)
            labels = labels.float().to(device=device)

            outputs = model(data)  # <4>

            loss = loss_fn(outputs, labels)  # <5>

            optimizer.zero_grad()  # <6>
            loss.backward()  # <7>
            optimizer.step()  # <8>

            running_loss += loss.item()  # <9>
            total += labels.size(0)
            correct += torch.round(outputs).eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        accu = 100. * correct / total

        train_accu.append(accu)
        train_losses.append(train_loss)
        valid_loss,valid_accu = test(model,valid_loader,device)
        print(
            f"Epoch': {epoch},Train loss: {train_loss},Train accuracy:{accu},Valid loss: {valid_loss},Valid accuracy: {valid_accu},Train MSE mean: {np.mean(train_losses)},Valid MSE mean: {np.mean(eval_losses)}")
        pbar.set_postfix({'Epoch': epoch, 'Train loss': train_loss, 'Train accuracy': accu, 'Valid loss': valid_loss,
                          'Valid accuracy': valid_accu, 'Train MSE mean': np.mean(train_losses),
                          'Valid MSE mean': np.mean(eval_losses)})
        if device.type != "cpu":
            pbar.set_postfix({'GPU memory allocated': torch.cuda.memory_allocated(device) / (1024 * 1024)})


def validate(model, train_loader, valid_loader, device):
    print(f"Validating on device {device}.")

    model.to(device=device)  # <2>
    model.eval()
    loss_fn = nn.MSELoss()

    for name, loader in [("train", train_loader), ("val", valid_loader)]:
        running_loss = 0.
        total = 0.
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(device=device)
                labels = labels.to(device=device).view(-1, 1).float()
                outputs = model(data)

                loss = loss_fn(outputs, labels)
                running_loss += loss.item()

        mse = running_loss / len(loader)
        print("MSE {}: {:.2f}".format(name, mse))


def evaluate(model, train_loader, valid_loader, device):
    print(f"Validating on device {device}.")

    model.to(device=device)  # <2>
    model.eval()

    result = {}
    for name, loader in [("train", train_loader), ("val", valid_loader)]:
        pred = torch.zeros(2,len(loader.dataset) // loader.batch_size + 1, loader.batch_size, model.nclasses)
        pred.to(device=device)
        with torch.no_grad():
            for count_loop, (data, labels) in enumerate(loader):
                data = data.to(device=device)
                labels = labels.float()
                outputs = model(data)
                pred[0, count_loop, :, :] = labels
                pred[1, count_loop, :, :] = outputs
        result[name] = pred.reshape(2, pred.shape[1]*pred.shape[2], model.nclasses).numpy()
    return result


def test(model, valid_loader, device):
    model.eval()

    running_loss = 0.
    correct = 0.
    total = 0.

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for data, labels in valid_loader:
            data = data.to(device=device)
            labels = labels.to(device=device).view(-1, 1).float()
            outputs = model(data)

            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

            total += labels.size(0)
            correct += torch.round(outputs).eq(labels).sum().item()

    valid_loss = running_loss / len(valid_loader)
    accu = 100. * correct / total

    eval_losses.append(valid_loss)
    eval_accu.append(accu)
    return valid_loss, accu

def density(args):
    device = (torch.device(args.device) if torch.cuda.is_available()
              else torch.device('cpu'))
    # if device.type=='cuda':
    #     mem = torch.cuda.get_device_properties(device).total_memory
    #     reserved = torch.cuda.memory_reserved(device)
    #     allocated = torch.cuda.memory_allocated(device)
    #     free = reserved - allocated  # free inside reserved
    #     print(f"Training on device {device},total GPU memory: {mem},allocated:{allocated} free:{free}, lr={args.lr:e}.")

    options = read_data_cfg(args.images)
    nclasses = int(options['classes'])
    train_path = pl.Path(options['train'])
    valid_path = pl.Path(options['valid'])

    model = DensityNet(1024, nclasses)

    # train loader
    train_label_path = pl.Path.joinpath(train_path.parent, train_path.stem + '_labels' + train_path.suffix)
    t1 = torch.load(train_path, map_location=device)
    t1 = t1.reshape(8862, 1792, 8, 10)
    t2 = torch.load(train_label_path, map_location=device)
    print(f"Loaded train features from {str(train_path)} - shape:{t1.shape}")
    print(f"Loaded train labels from {str(train_label_path)} - shape:{t2.shape}")
    trainset = list(zip(t1, t2))

    # trainset = dataset.featureDataset(train_path, shape=(10, 8),
    #                                        shuffle=False,
    #                                        )

    train_loader = D.DataLoader(trainset, batch_size=args.bs,
                                shuffle=True)
    # test loader
    valid_label_path = pl.Path.joinpath(valid_path.parent, valid_path.stem + '_labels' + valid_path.suffix)
    v1 = torch.load(valid_path, map_location=device)
    v1 = v1.reshape(1366, 1792, 8, 10)
    v2 = torch.load(valid_label_path, map_location=device)
    print(f"Loaded train features from {str(valid_path)} - shape:{v1.shape}")
    print(f"Loaded train labels from {str(valid_label_path)} - shape:{v2.shape}")
    valset = list(zip(v1, v2))

    # valset = dataset.featureDataset(valid_path, shape=(10, 8),
    #                                   shuffle=False,
    #                                   )
    valid_loader = D.DataLoader(valset, batch_size=args.bs,
                                shuffle=False)

    # retrieve weights
    # model.load_state_dict(torch.load('D:/results/10_18_2conv_norm.pt',map_location="cpu"))
    train(model, args, train_loader, valid_loader, device)
    # if args.save:
    dateTimeObj = datetime.now()
    outdir = pl.Path.joinpath(train_path.parent, str(dateTimeObj.year) + '_' + \
                              str(dateTimeObj.month) + '_' + \
                              str(dateTimeObj.day) + '_' + str(dateTimeObj.hour) + '_' + \
                              str(dateTimeObj.minute) + '_' + str(dateTimeObj.second))
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), pl.Path.joinpath(outdir, 'trained_model.pt'))
    validate(model, train_loader=train_loader, valid_loader=valid_loader, device=device)
    #predictions = evaluate(model, train_loader, valid_loader, device)

    fig = plt.figure()
    plt.plot(train_losses)
    plt.plot(eval_losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend(['Train', 'Valid'])
    plt.title('Train vs Valid Losses')
    plt.savefig(pl.Path.joinpath(outdir, 'train_loss.png'))
    # plt.show()

    return model

if __name__ == '__main__':
    args = cmdline.arg_parse()
    # 1. get tensor
    # 2. evaluation
    # tensor = torch.load(tensorpath,map_location=torch.device("cpu")).reshape(8862,1792,8,10)
    # print(tensor)
    # m = DensityNet()
    # m.eval()
    # output = m(tensor[0].  (0))
    # print(output)
    # 3. training
    model = density(args)
    # model = DensityNet()
    # model.load_state_dict(torch.load('D:/weights/10_8_avg_simple_model.pt', map_location="cpu"),strict=False)
    # options = read_data_cfg('data/kaist_density.data')
    # train_path = pl.Path(options['train'])
    # valid_path = pl.Path(options['valid'])
    #
    # device = (torch.device('cuda') if torch.cuda.is_available()
    #           else torch.device('cpu'))
    # # train loader
    # train_label_path = pl.Path.joinpath(train_path.parent, train_path.stem + '_labels' + train_path.suffix)
    # t1 = torch.load(train_path, map_location=device)
    # t1 = t1.reshape(8862, 1792, 8, 10)
    # t2 = torch.load(train_label_path, map_location=device).reshape(8862)
    # print("Loaded train features from {},labels from {}".format(str(train_path), str(train_label_path)))
    # trainset = list(zip(t1, t2))
    # train_loader = D.DataLoader(trainset, batch_size=64,
    #                             shuffle=True)
    # valid_label_path = pl.Path.joinpath(valid_path.parent, valid_path.stem + '_labels' + valid_path.suffix)
    # v1 = torch.load(valid_path, map_location=device)
    # print(v1.shape)
    # v1 = v1.reshape(1366, 1792, 8, 10)
    # v2 = torch.load(valid_label_path, map_location=device).reshape(1366)
    # print("Loaded lid features from {},labels from {}".format(str(valid_path), str(valid_label_path)))
    # valset = list(zip(v1, v2))
    # valid_loader = torch.utils.data.DataLoader(valset, batch_size=64,
    #                                            shuffle=False)
    # predictions = evaluate(model, None, train_loader, valid_loader, device)
    #
    # train_predict = np.array(predictions['train'])
    # train_predict = train_predict.reshape(2, train_predict.shape[1]).transpose()
    # eval_predict = np.array(predictions['val'])
    # eval_predict = eval_predict.reshape(2, eval_predict.shape[1]).transpose()
    #
    # np.save(pl.Path.joinpath(train_path.parent, 'mse_train_arr'), train_predict)
    # np.save(pl.Path.joinpath(train_path.parent, 'mse_valid_arr'), eval_predict)
