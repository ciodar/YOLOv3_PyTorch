import sys
import time
from PIL import Image, ImageDraw
#from models.tiny_yolo import TinyYoloNet
import pathlib as pl
import cmdline
from utils import *
from image import letterbox_image, correct_yolo_boxes
from darknet import Darknet
import cv2
import tqdm

namesfile=None

def detect_model(args):

    images = pl.Path(args.images)
    m = Darknet(args.cfgfile)

    check_model = args.cfgfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(args.cfgfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(args.weightsfile)

    # m.print_network()
    cuda = False
    if cuda:
        m.cuda()
        cuda_device='cuda:0'
    else:
        cuda_device='cpu'

    m.eval()

    class_names = load_class_names(args.namesfile)
    newdir = pl.Path.joinpath(images,'predicted')
    if not os.path.exists(newdir):
        os.mkdir(newdir)

    start = time.time()
    total_time = 0.0
    # count_img = 0
    for count_img, imgfile in enumerate(tqdm.tqdm(list(pl.Path(images).rglob('*.png')))):
        # count_img +=1

        img = cv2.imread(str(imgfile))
        sized = cv2.resize(img, (m.width, m.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        detect_time_start = time.time()
        boxes = do_detect(m, sized, args.confidence, args.nms_thresh, cuda)

        detect_time_end = time.time() - detect_time_start
        total_time += detect_time_end

        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        plot_boxes_cv2(img, boxes, class_names=class_names, color=red)

        savename = newdir.joinpath(imgfile.name)
        # print("save plot results to %s" % savename)
        cv2.imwrite(str(savename), img)
    finish = time.time() - start

    count_img += 1
    print('len dir = %d ' % (count_img))
    # print('Predicted in %d minutes %f seconds with average %f seconds / image.' % (finish//60, finish%60, finish/count_img))
    print('Predicted in %d minutes %f seconds with average %f seconds / image.' % (
    finish // 60, finish % 60, total_time / count_img))


def detect_cv2(args):

    m = Darknet(args.cfgfile)
    m.print_network()
    m.load_weights(args.weightsfile)
    print('Loading weights from %s... Done!' % (args.weightsfile))
    
    cuda = False
    if cuda:
        m.cuda()

    img = cv2.imread(args.images)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    
    start = time.time()
    boxes = do_detect(m, sized, args.confidence, args.nms_thresh, cuda)
    finish = time.time()

    class_names = load_class_names(args.namesfile)
    print(len(boxes))
    plot_boxes_cv2(img, boxes, class_names=class_names)
    savename = args.images.split('.')[0]
    savename = savename+'_predicted.jpg'
    print("save plot results to %s" % savename)
    cv2.imwrite(savename, img)

def readvideo_cv2(args):
    m = Darknet(args.cfgfile)
    # m.print_network()
    m.load_weights(args.weightsfile)
    print('Loading weights from %s... Done!' % (args.weightsfile))
    cuda=False
    if cuda:
        m.cuda()

    cap = cv2.VideoCapture(args.images)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('result_' + args.images, fourcc, 28, (frame_width, frame_height))
    start = time.time()
    count_frame = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            count_frame += 1
            # Display the resulting frame
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sized = cv2.resize(frame, (m.width, m.height))

            # print('shape 1: ')
            # print(sized.shape)


            new_img = np.zeros_like(sized)
            img_mean = np.mean(sized,-1)
            new_img[:,:,0] = img_mean
            new_img[:,:,1] = img_mean
            new_img[:,:,2] = img_mean

            sized = new_img

            boxes = do_detect(m, sized, args.confidence, args.nms_thresh, cuda)

            class_names = load_class_names(namesfile)

            ##add this
            frame = new_img

            frameResult = plot_boxes_cv2(frame, boxes, class_names=class_names)

            cv2.imshow('Frame', frameResult)

            cv2.imwrite('./carstops/img%06d.jpg'%(count_frame),frameResult)
            out.write(frameResult)

            # Press Q on keyboard to  exit
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        else:
            break
    finish = time.time()
    print('Processed video %s with %d frames in %f seconds.' % (args.images, count_frame, (finish - start)))
    print("Saved video result to %s" % ('result_' + args.images))
    cap.release()
    out.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    args = cmdline.arg_parse()
    images = args.images
    batch_size = int(args.bs)
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    # if len(sys.argv) >= 1:
    #     if len(sys.argv) == 2:
    #         imgfile = sys.argv[1]
    #     elif len(sys.argv) == 3:
    #         imgfile = sys.argv[1]
    #         weightfile = sys.argv[2]

    if os.path.isdir(images):
        detect_model(args)
    elif (images.split('.')[1] == 'jpg') or (images.split('.')[1] == 'png') or (images.split('.')[1] == 'jpeg'):
        detect_cv2(args)
    else:
        readvideo_cv2(args)
    # else:
    #     print('Usage: ')
    #     print('  python detect.py image/video/folder [weightfile]')
    #     print('  or using:  python detect.py thermal_kaist.png ')
