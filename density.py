import cmdline

def density(args):
    m = Darknet(args.cfgfile)
    check_model = args.cfgfile.split('.')[-1]
    if check_model == 'model':
        checkpoint = torch.load(args.cfgfile)
        # print('Load model from ', modelfile)
        m.load_state_dict(checkpoint['state_dict'])
    else:
        m.load_weights(args.weightsfile)
    m.print_network()

if __name__ == '__main__':
    args = cmdline.arg_parse()