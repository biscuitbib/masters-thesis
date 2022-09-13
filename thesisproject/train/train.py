import os
import torch
from torch import Tensor, nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import pad
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from skimage.io import imread, imsave
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from thesisproject.utils import get_metrics, mask_to_rgb, segmentation_to_rgb, grayscale_to_rgb

def training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=10, cont=False):
    writer = SummaryWriter()
    print(writer)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = os.path.join("model_saves", "model_checkpoint.pt")
    model_path = os.path.join("model_saves", "model.pt")
    start_epoch = 0

    if cont:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    net.train()

    for epoch in range(start_epoch, num_epochs):  # loop over the dataset multiple times

        # Training loop
        pbar = tqdm(total=len(train_loader), position=0, leave=True)
        train_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print(outputs.shape, outputs.dtype)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            batch_samples = inputs.shape[0]
            current_loss = loss.item() / batch_samples
            train_loss += current_loss

            num_batches += 1

            #pbar.update(inputs.shape[0])
            pbar.update(1)

        pbar.set_description(f"training loss epoch {epoch}: {round(train_loss/num_batches, 3)}")

        # Validation
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_specificity = 0.0
            val_dice = 0.0

            num_val_batches = 0

            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                # save statistics
                batch_samples = inputs.shape[0]
                current_loss = loss.item() / batch_samples
                metrics = get_metrics(outputs, labels, remove_bg=True)

                val_loss += current_loss
                val_accuracy += metrics["accuracy"]
                val_precision += metrics["precision"]
                val_recall += metrics["recall"]
                val_specificity += metrics["specificity"]
                val_dice += metrics["dice"]

                num_val_batches += 1

            # Write to tensorboard
            input_imgs = inputs.cpu() * 255
            input_imgs /= torch.max(input_imgs)
            input_imgs = grayscale_to_rgb(input_imgs)
            pred_imgs = segmentation_to_rgb(outputs.cpu())
            target_imgs = mask_to_rgb(labels.cpu())
            imgs = torch.cat((input_imgs, target_imgs, pred_imgs), dim=2)

            writer.add_images("images/val", imgs[:4, ...], epoch)
            #writer.add_images("images/val_inputs", input_imgs[:4, ...], epoch)
            #writer.add_images("images/val_targets", target_imgs[:4, ...], epoch)
            #writer.add_images("images/val_predictions", pred_imgs[:4, ...], epoch)

            writer.add_scalar("validation_metrics/accuracy", val_accuracy/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/precision", val_precision/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/recall", val_recall/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/specificity", val_specificity/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/dice", val_dice/num_val_batches, epoch)

            writer.add_scalar("loss/train", train_loss/num_batches, epoch)
            writer.add_scalar("loss/validation", val_loss/num_val_batches, epoch)

        # Save model checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)


    # Save final model
    torch.save(model.state_dict(), model_path)

    print('Finished Training')

def train(net, train_loader, val_loader):
    logger = TensorBoardLogger("runs", name="unet")
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[EarlyStopping(monitor="loss/validation", mode="min")],
        profiler="simple"
    )
    trainer.fit(
        model=net,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )