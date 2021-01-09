from torch import nn
import torch.utils.data.DataLoader as DataLoader
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

invTrans = transforms.Compose([ 
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
])


class DemonstrationDataset(data.Dataset):
    def __init__(self, root, train, transform, input_size):
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.obs_dir = os.path.join(root, 'obs')
        self.seg_dir = os.path.join(root, 'seg')
        self.action_fname = os.path.join(root, 'action.txt')
        self.length = 5 # length of frame sequence into the future

        obs_list = os.listdir(self.obs_dir)
        seg_list = os.listdir(self.seg_dir)
        action_f = open(self.action_fname, 'r')

        self.epoch_dict = {}
        for fname in self.obs_list:
            fname = fname.split('.jpg')[0]
            items = fname.split('_')
            epoch, step = int(items[0]), int(items[1])
            if epoch in self.epoch_dict.keys():
                self.epoch_dict[epoch] = max(step, self.epoch_dict[epoch])
            else:
                self.epoch_dict[epoch] = step

        self.action_dict = {}
        lines = action_f.readlines()
        for line in lines:
            items = line.strip().split()
            fname, thr, steer = items[0], float(items[1]), float(items[2])
            items = fname.split('.jpg')[0]
            items = items.split('_')
            epoch, step = int(items[0]), int(items[1])
            assert step <= self.epoch_dict[epoch]
            if epoch in self.action_dict.keys():
                self.action_dict[epoch][step] = [thr, steer]
            else:
                self.action_dict[epoch] = {}
                self.action_dict[epoch][step] = [thr, steer]

        self.n_samples = 0
        self.epoch_start_indices = []
        for epoch in self.epoch_dict.keys():
            # by default, the model predicts at more 10 steps into future
            self.epoch_start_indices.append([epoch, self.n_samples])
            self.n_samples += self.epoch_dict[epoch] - 10
        self.n_epoch = len(self.epoch_start_indices)
        
    def readImg(self, fname):
        img = Image.open(fname)
        img = self.transform(img)
        return img
    
    def __getitem__(self, idx):
        for i in range(self.n_epoch - 1):
            if i == self.n_epoch - 2:
                epoch = self.epoch_start_indices[-1][0]
                step = self.n_samples - self.epoch_start_indices[-1][1]
                break 
            if idx >= self.epoch_start_indices[i][1] and idx < self.epoch_start_indices[i+1][1]:
                # sampled from the ith epoch
                epoch = self.epoch_start_indices[i][0]
                step = idx - self.epoch_start_indices[i][1]
                break
        
        obs_fname = os.path.join(self.obs_dir, '{}_{}.jpg'.format(epoch, step))
        seg_fname = os.path.join(self.seg_dir, '{}_{}.npy'.format(epoch, step))
        action = self.action_dict[epoch][step]

        img = self.readImg(obs_fname)

        seg = np.load(seg_fname)
        seg = torch.from_numpy(seg)

        action_var = torch.from_numpy(np.array([-1.0, 0.0])).repeat(1, 2, 1)
        obs_var = img.repeat(1, 2, 1)
        if step == 1:
            action_var[-1] = self.action_dict[epoch][step-1]
            obs_fname = os.path.join(self.obs_dir, '{}_{}.jpg'.format(epoch, step-1))
            obs_var[-1] = self.readImg(obs_fname)
        if step > 1:
            action_var[-1] = self.action_dict[epoch][step-1]
            action_var[-2] = self.action_dict[epoch][step-2]
            obs_fname = os.path.join(self.obs_dir, '{}_{}.jpg'.format(epoch, step-1))
            obs_var[-1] = self.readImg(obs_fname)
            obs_fname = os.path.join(self.obs_dir, '{}_{}.jpg'.format(epoch, step-2))
            obs_var[-2] = self.readImg(obs_fname)

        return img, seg, action, action_var, obs_var

    def __len__(self):
        return self.n_samples


def train(args):
    guides = generate_guide_grid(args.bin_divide)
    optimizer = optim.Adam(train_net.parameters(), lr=args.lr, amsgrad=True)

    # initialize the training model
    train_net = ConvLSTMMulti(args)
    train_net.train()
    if torch.cuda.is_available():
        train_net = train_net.cuda()
        if args.data_parallel:
            train_net = torch.nn.DataParallel(train_net)

    trainset = DemonstrationDataset('Demonstrations', train=True, transform=transform, input_size=256)
    testset = DemonstrationDataset('Demonstrations', train=False, transform=transform, input_size=256)

    trainloader = DataLoader(trainset, 20, shuffle=True, num_workers=16)
    testloader = DataLoader(testset, 20, shuffle=False, num_workes=16)

    for batch_idx, imgs, segs, actions, action_vars, obs_vars in enumerate(trainloader):

    


def eval(args):

if __name__ == '__main__':
    main()