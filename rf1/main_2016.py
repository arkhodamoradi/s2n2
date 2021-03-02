import argparse
from utils import *
from network2016 import *
import datetime
from torch.utils.tensorboard import SummaryWriter
from loader_2016 import get_radio_ml_loader_2016 as get_loader2016
from loader_2016 import get_numpy_2016 as get_lists
from data.data_utils import iq2spiketrain as to_spike_train
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='RFSNN')
    parser.add_argument('--radio_ml_data_dir', type=str, default='2018.01',
                        help='path to the folder containing the RadioML HDF5 file(s)')
    parser.add_argument('--min-snr', type=int, default=6,
                        metavar='N', help='minimum SNR (inclusive) to use during data loading')
    parser.add_argument('--max-snr', type=int, default=30,
                        metavar='N', help='maximum SNR (inclusive) to use during data loading')
    parser.add_argument('--labels', type=int, default=24,
                        metavar='N', help='number of targets')
    parser.add_argument('--intervals', type=int, default=64,
                        metavar='N', help='sub time windows for grad calculation (0, 1024]')
    parser.add_argument('--per-h5-frac', type=float, default=0.5,
                        metavar='N', help='fraction of each HDF5 data file to use')
    parser.add_argument('--per-sample-frac', type=float, default=1.0,
                        metavar='N', help='fraction of each sample to use (1 = 1024)')
    parser.add_argument('--train-frac', type=float, default=0.9,
                        metavar='N', help='train split (1-TRAIN_FRAC is the test split)')
    parser.add_argument('--IQ-resolution', type=int, default=128,
                        metavar='N', help='size of I/Q dimension (used when representing I/Q plane as image)')
    parser.add_argument('--seed', type=int, default=123,
                        metavar='S', help='random seed')
    parser.add_argument('--epochs', default=500, type=int, metavar='N',
                        help='number of total epochs')
    parser.add_argument('--batch-size', default=24, type=int, metavar='N',
                        help='mini-batch size (default: 25)')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--optim', default='SGD', type=str, metavar='OPTIM',  # TODO: ADAM
                        help='optimizer (default: SGD)')
    parser.add_argument('--loss-func', default='CrossEntropyLoss', type=str, metavar='OPTIM',
                        help='optimizer (default: CrossEntropy)')
    parser.add_argument('--save-path', default='runs2016', type=str, metavar='path',
                        help='Result path')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--synapse-reset', dest='synapse', action='store_true',
                        help='Synapse reset mechanism')
    parser.add_argument('--skip-1', dest='dataLoader', action='store_true',
                        help='skip every other sample')

    return parser.parse_args()


def get_kwargs(args):
    to_st_kwargs = {}
    to_st_kwargs['out_w'] = args.IQ_resolution
    to_st_kwargs['out_h'] = args.IQ_resolution

    return to_st_kwargs


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()

    args.radio_ml_data_dir = '/mnt/013c8c34-4de2-4dab-9e29-16618f093336/playground/RFSNN/RML2016.10a/RML2016.10a_dict.pkl'
    args.epochs = 8
    args.min_snr = -20
    args.max_snr = 18
    args.labels = 11
    args.per_h5_frac = 1.0  # (0.5)
    args.per_sample_frac = 1.0  # (1.0) * 1024
    args.train_frac = 0.9
    args.IQ_resolution = 16
    args.batch_size = 24
    args.intervals = 128
    args.lr = 0.1
    args.synapse_reset = True
    args.skip_1 = False
    args.loss_func = 'SmoothL1Loss'  # 'SmoothL1Loss'  'CrossEntropyLoss'
    # threshs = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.2, 0.1]
    # threshs = [0.5, 0.3, 0.3, 0.3, 0.3, 0.3]
    threshs = [0.3, 0.2, 0.3, 0.3]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(args.save_path, current_time)
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    to_st_kwargs = get_kwargs(args)

    X_train, X_test, Y_train, Y_test = get_lists(data_dir=args.radio_ml_data_dir, min_snr=args.min_snr,max_snr=args.max_snr, train_frac=args.train_frac)
    train_loader = get_loader2016(args.batch_size, X_train, Y_train, train=True, normalize=True)
    test_loader = get_loader2016(args.batch_size, X_test, Y_test, train=False, normalize=True)

    # Model
    model = Spatial_CNN(nb_epochs=args.epochs,
                        lr=args.lr,
                        resume=args.resume,
                        start_epoch=args.start_epoch,
                        evaluate=args.evaluate,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        optim=args.optim,
                        crit=args.loss_func,
                        target_size=args.labels,
                        intervals=args.intervals,
                        threshs=threshs,
                        writer=writer,
                        synapse_reset=args.synapse_reset,
                        **to_st_kwargs)

    # Training
    model.run()


class Spatial_CNN():
    def __init__(self, nb_epochs, lr, resume, start_epoch, evaluate, train_loader, test_loader,
                 optim, crit, target_size, intervals, threshs, writer, synapse_reset, **kwargs):
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.resume = resume
        self.start_epoch = start_epoch
        self.evaluate = evaluate
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.best_prec1 = 0
        self.target_size = target_size
        self.intervals = intervals
        self.kwargs = kwargs
        self.writer = writer
        self.cur_epoch = 0
        self.train_iter = 0
        self.test_iter = 1
        self.layer_names = []
        self.grad_idx = 0

        data0, label = next(iter(self.train_loader))

        input_spikes = to_spike_train(data0, **self.kwargs)
        self.input_size = input_spikes.size()
        print('==> Build model and setup loss and optimizer')
        self.model = spiking_resnet_18(self.input_size, synapse_reset=synapse_reset, threshs=threshs,
                                       nb_classes=self.target_size).cuda()
        self.criterion = getattr(nn, crit)().cuda()
        if optim == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9, nesterov=True)
        elif optim == 'ADAM':
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        else:
            raise print('{} optimizer is not acceptable. Please select SGD or ADAM'.format(optim))

    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                # self.best_prec1 = checkpoint['best_prec1']
                self.best_prec1 = 30
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                      .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_1epoch()
            return

    def get_layer_names(self):
        for n, p in self.model.named_parameters():
            l_name = n.split('.')[0]
            if not (l_name in self.layer_names):
                self.layer_names.append(l_name)

    def set_grads_requirements(self):
        # we loop from end of the list for the length of it
        _len = len(self.layer_names)
        _idx = _len - 1 - self.grad_idx % _len
        _name = self.layer_names[_idx]
        for n, p in self.model.named_parameters():
            if n.split('.')[0] == _name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def run(self):
        self.resume_and_evaluate()
        cudnn.benchmark = True
        # self.get_layer_names()

        for self.epoch in range(self.start_epoch, self.nb_epochs):
            self.train_1epoch()
            prec1 = self.validate_1epoch()
            is_best = prec1 > self.best_prec1
            # save model
            save_checkpoint({
                'epoch': self.epoch,
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_prec1,
                'optimizer': self.optimizer.state_dict()
            }, is_best, 'record2016/spatial/checkpoint.pth.tar', 'record2016/spatial/model_best.pth.tar')
            self.cur_epoch += 1

    def record_grads(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.writer.add_histogram('Grads/' + n, p.grad, self.train_iter)
                self.writer.add_histogram('Weights/' + n, p, self.train_iter)

    def to_one_hot(self, t, width):
        if str(self.criterion) == 'SmoothL1Loss()':
            t_onehot = torch.zeros(*t.shape + (width,))
            return t_onehot.scatter_(1, t.unsqueeze(-1), 1)
        elif str(self.criterion) == 'CrossEntropyLoss()':
            return t
        else:
            raise print('Not ready for ' + str(self.criterion))

    def train_1epoch(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to train mode
        self.model.train()
        end = time.time()
        # mini-batch training
        progress = tqdm(self.train_loader)
        for i, (data0, label) in enumerate(progress):
            # self.set_grads_requirements()
            self.grad_idx += 1
            # measure data loading time
            data_time.update(time.time() - end)

            input_spikes = to_spike_train(data0, **self.kwargs)
            label_one_hot = self.to_one_hot(label, self.target_size)

            window_blocks = self.input_size[1] // self.intervals
            if input_spikes.shape[0] != self.input_size[0]:
                continue  # avoiding data less than batch size
            reset = True
            # cur_lr = lr_step_size = self.lr / window_blocks
            for wb in range(window_blocks):
                output, rs = self.model(input_spikes[:, wb * self.intervals:(wb + 1) * self.intervals].cuda(),
                                        self.intervals, reset)

                sn = 0
                for r in rs:
                    self.writer.add_scalar('Spikes/' + str(sn), r, self.train_iter)
                    sn += 1

                reset = False
                loss = self.criterion(output, label_one_hot.cuda())
                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.2)
                self.record_grads()
                # This line is used to prevent the vanishing / exploding gradient problem
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                # self.optimizer.param_groups[0]['lr'] = cur_lr  # TODO: check with nesterov and wd
                self.optimizer.step()
                # cur_lr += lr_step_size
                self.train_iter += 1

            # --- measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, label.cuda(), topk=(1, 5))
            losses.update(loss.item(), data0.size(0))
            top1.update(prec1.item(), data0.size(0))
            top5.update(prec5.item(), data0.size(0))
            self.writer.add_scalar('Train/loss(step)', loss.item(), self.train_iter // window_blocks)
            self.writer.add_scalar('Train/top1(step)', prec1.item(), self.train_iter // window_blocks)
            self.writer.add_scalar('Train/top5(step)', prec5.item(), self.train_iter // window_blocks)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Data Time': [round(data_time.avg, 3)],
                'Loss': [round(losses.avg, 5)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                'lr': self.optimizer.param_groups[0]['lr']
                }
        record_info(info, 'record2016/spatial/rgb_train.csv', 'train')

    def validate_1epoch(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        progress = tqdm(self.test_loader)
        with torch.no_grad():
            for i, (data0, label) in enumerate(progress):

                input_spikes = to_spike_train(data0, **self.kwargs)

                if input_spikes.shape[0] != self.input_size[0]:
                    continue  # avoiding data less than batch size

                output, r = self.model(input_spikes.cuda(), self.input_size[1], True)
                # compute output
                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # Calculate pred acc
                prec1, prec5 = accuracy(output.data, label.cuda(), topk=(1, 5))
                top1.update(prec1.item(), data0.size(0))
                top5.update(prec5.item(), data0.size(0))
                self.writer.add_scalar('Eval/top1(step)', prec1.item(), self.test_iter)
                self.writer.add_scalar('Eval/top5(step)', prec5.item(), self.test_iter)
                self.test_iter += 1

        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Prec@1': [round(top1.avg, 4)],
                'Prec@5': [round(top5.avg, 4)],
                }
        record_info(info, 'record2016/spatial/rgb_test.csv', 'test')
        return top1.avg


if __name__ == '__main__':
    main()
