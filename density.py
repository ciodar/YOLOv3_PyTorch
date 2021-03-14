import cmdline
from darknet import Darknet
from utils import image2torch
import cv2
import torch

def density(args):
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
    use_cuda = torch.cuda.is_available() and (True if use_cuda is None else use_cuda)
    m.eval()

    img = image2torch(img)
    img = img.to(torch.device("cuda" if use_cuda else "cpu"))
    out_boxes = m(img)
    print(out_boxes)


if __name__ == '__main__':
    args = cmdline.arg_parse()
    density(args)