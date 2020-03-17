import torch
import torch.utils.data as data

def get_loader(args, is_train):
  args.dataset = args.dataset.lower()

  if args.dataset == 'cifar10':
    dset = get_cifar(is_train)
  elif args.dataset == 'Gaussian1D':
    dset = Gaussian1D(args.fdata)
  else:
    raise NotImplementedError('Currently only supporting CIFAR10. \n Ciao~! :)')

  if is_train and args.use_val:
    n_train = int(0.8 * len(dset))
    n_val = len(dset) - n_train
    dset_train, dset_val = torch.utils.data.random_split(dset, [n_train, n_val])
    return data.DataLoader(dset_train, args.bt, shuffle=True, num_workers=args.num_workers), \
           data.DataLoader(dset_val, args.bt, shuffle=False, num_workers=args.num_workers)

  return data.DataLoader(dset, args.bt, shuffle=is_train)

def get_cifar(is_train):
    if is_train:
      # No normalization applied, since Glow expects inputs in (0, 1)
      transform_train = transforms.Compose([
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor()
      ])
      trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
      return trainset
    else:
      transform_test = transforms.Compose([
          transforms.ToTensor()
      ])
      testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
      return testset

class Gaussian1D(data.Dataset):
  def __init__(self, fdata):
    self.X = np.load(fdata)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx]
