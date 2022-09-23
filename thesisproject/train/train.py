import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.io import imread, imsave
from tqdm import tqdm
import threading

from thesisproject.utils import get_metrics, mask_to_rgb, segmentation_to_rgb, grayscale_to_rgb

def training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=10, cont=False):
    writer = SummaryWriter()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    # Load checkpoint if continuing
    checkpoint_path = os.path.join("model_saves", "model_checkpoint.pt")
    model_path = os.path.join("model_saves", "model.pt")
    start_epoch = 0

    if cont:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Continuing training from epoch {start_epoch}")


    # Training
    net.train()

    for epoch in range(start_epoch, num_epochs):
        pbar = tqdm(total=len(train_loader) + len(val_loader), position=0, leave=True)
        pbar.set_description(f"Epoch {epoch} training")
        train_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
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
        
        pbar.set_description(f"Epoch {epoch} validation")
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
                metrics = get_metrics(outputs.detach().cpu(), labels.detach().cpu(), net.n_classes, remove_bg=True)

                val_loss += current_loss
                val_accuracy += metrics["accuracy"]
                val_precision += metrics["precision"]
                val_recall += metrics["recall"]
                val_specificity += metrics["specificity"]
                val_dice += metrics["dice"]

                num_val_batches += 1
                
                pbar.update(1)

            # Write to tensorboard
            input_imgs = inputs.cpu() * 255
            input_imgs /= torch.max(input_imgs)
            input_imgs = grayscale_to_rgb(input_imgs)
            pred_imgs = segmentation_to_rgb(outputs.cpu())
            pred_overlay = (input_imgs / 2) + (pred_imgs / 2)
            
            target_imgs = mask_to_rgb(labels.cpu())
            target_overlay = (input_imgs / 2) + (target_imgs / 2)
            imgs = torch.cat((target_overlay, pred_overlay), dim=2)

            writer.add_images("images/val", imgs[:4, ...], epoch)
            #writer.add_images("images/val_inputs", input_imgs[:4, ...], epoch)
            #writer.add_images("images/val_targets", target_imgs[:4, ...], epoch)
            #writer.add_images("images/val_predictions", pred_imgs[:4, ...], epoch)

            writer.add_scalar("validation_metrics/accuracy", val_accuracy/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/precision", val_precision/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/recall", val_recall/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/specificity", val_specificity/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/dice", val_dice/num_val_batches, epoch)

            writer.add_scalar("loss/validation", val_loss/num_val_batches, epoch)

            writer.add_scalar("loss/train", train_loss/num_batches, epoch)
            
            writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

        # Save model checkpoints
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, checkpoint_path)

        # Step learning rate scheduler
        scheduler.step(val_dice/num_val_batches)
        
        print(f"training loss epoch {epoch}: {round(train_loss/num_batches, 3)}")

    # Save final model
    torch.save(net.state_dict(), model_path)
    pbar.close()

    print('Finished Training')