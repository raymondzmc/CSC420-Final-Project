import torch, os, pdb
import config as cfg
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import FashionNet
from dataset import FashionDataset
from eval import eval_fashion


def train(last_ckpt=None, task='person'):
    cuda = torch.cuda.is_available()

    # Directory for saving model checkpoints
    ckpt_path = os.path.join(cfg.ckpt_path, task)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)

    n_classes = 1 if task == 'person' else 7
    model = FashionNet(n_classes=n_classes)

    transform = transforms.Compose([
                    transforms.Resize(cfg.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=cfg.mean, std=cfg.std)])

    # Create dataloader for training the model
    train_set = FashionDataset(cfg.data_path, cfg.label_path, transform, mode=task)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, pin_memory=cuda)
    test_set = FashionDataset(cfg.data_path, cfg.label_path, transform, mode=task, train=False)
    test_loader = DataLoader(test_set, batch_size=1, pin_memory=cuda)

    # Use a simple Binary Cross-Entropy loss as criterion
    criterion = torch.nn.BCEWithLogitsLoss()

    # Specify ADAM optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.decay)

    # Reduce the training rate by a factor of 0.5 every 50 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Move model to GPU if available
    if cuda:
        model = model.cuda()

    # Load the last checkpoint if specified
    if last_ckpt is not None:
        model.load_state_dict(torch.load(os.path.join(ckpt_path, "{}.pth".format(last_ckpt))))
        print("Loaded checkpoint at epoch {}".format(last_ckpt))
    else:
        last_ckpt = 1

    # Train the model for epochs defined in config
    for epoch in range(last_ckpt, cfg.epochs + 1):
        batch_loss = 0.

        # Iterate over the dataloader for the training set
        for i, (batch_input, batch_target) in enumerate(train_loader):

            if cuda:
                batch_input, batch_target = batch_input.cuda(), batch_target.cuda()

            batch_input, batch_target = Variable(batch_input), Variable(batch_target)
            optimizer.zero_grad()
            batch_output = model(batch_input)

            loss = criterion(batch_output, batch_target)
            # print("Loss for Minibatch {} of {}: {:.5f}".format(i + 1, len(train_loader), loss.item()))
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        print("Loss for Batch {} of {}: {:.5f}".format(epoch, cfg.epochs, batch_loss))

        # Save state dict
        torch.save(model.state_dict(), os.path.join(ckpt_path, '{}.pth'.format(epoch)))


if __name__ == "__main__":
    # train(last_ckpt=500, task='person')
    train(task='clothes')