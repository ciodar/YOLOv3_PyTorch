import argparse
def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument('images',metavar = 'images', help =
    "Image / Directory containing images to perform detection upon", type = str)
    parser.add_argument("--det", dest = 'det', help =
    "Image / Directory to store detections to",
                        default = "K:/output", type = str)
    parser.add_argument("--set", dest='set', help=
    "Subset of the dataset to be evaluated (train/valid/test)",
                        default="train", type=str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1,type=int)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help ="Config file",default = "cfg/yolov3_flir.cfg", type = str)
    parser.add_argument("--data", dest='datacfg', help="Data configuration file", default="data/flir.data", type=str)
    parser.add_argument("--weights", dest = 'weightsfile', help ="weightsfile", default = "K:/weights/flir_detector.weights",
                        type = str)
    parser.add_argument("--reso", dest = 'reso',
                        help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    parser.add_argument("--names", dest = 'namesfile',
                        help = "Names file",
                        default = "data/flir.names", type = str)
    parser.add_argument("--cuda", dest = 'cuda',
                        help = "Enable the use of the GPU",
                        default = False, type = bool)
    parser.add_argument("--device", dest = 'device',
                        help = "Choose the device to be used",
                        default = "cuda:0", type = str)
    parser.add_argument('--epoch', '-e', type=int, default=50,help='How many epoch we train, default is 30')
    parser.add_argument('--lr', type=float, default=1e-3, help='How many epoch we train, default is 30')
    return parser.parse_args()