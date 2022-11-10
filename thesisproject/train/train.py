import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from time import time

from thesisproject.utils import get_multiclass_metrics, create_overlay_figure, create_confusion_matrix_figure

def training_loop(net, criterion, optimizer, train_loader, val_loader, num_epochs=10, cont=False, model_name="model"):
    layout = {
        "Loss": {"loss": ["Multiline", ["loss/train", "loss/validation"]]},
        "Per class dice": {"dice": ["Multiline", [f"dice/{name}" for name in net.class_names]]}
    }

    writer = SummaryWriter()
    writer.add_custom_scalars(layout)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)

    # Load checkpoint if continuing
    checkpoint_path = os.path.join("model_saves", f"{model_name}_checkpoint.pt")
    model_path = os.path.join("model_saves", f"{model_name}.pt")
    start_epoch = 0

    if cont:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Continuing training from epoch {start_epoch}")


    # Training
    for epoch in range(start_epoch, num_epochs):
        start_time = time()
        net.train()
        #pbar = tqdm(total=len(train_loader) + len(val_loader), position=0, leave=True)
        #pbar.set_description(f"Epoch {epoch} training")
        print(f"Epoch {epoch} ", end="")
        train_loss = 0.0
        num_batches = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            batch_samples = inputs.shape[0]
            current_loss = loss.item() / batch_samples
            train_loss += current_loss

            num_batches += 1

            #pbar.update(inputs.shape[0])
            #pbar.update(1)

        #pbar.set_description(f"Epoch {epoch} validation")

        net.eval()
        # Validation
        with torch.no_grad():
            val_loss = 0.0
            val_accuracy = 0.0
            val_precision = 0.0
            val_recall = 0.0
            val_specificity = 0.0
            val_dice = 0.0
            val_per_class_dice = np.zeros(net.n_classes - 1)

            num_val_batches = 0

            for i, data in enumerate(val_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = net(inputs)

                loss = criterion(outputs, labels)

                # save statistics
                batch_samples = inputs.shape[0]
                current_loss = loss.item() / batch_samples
                metrics = get_multiclass_metrics(outputs.detach().cpu(), labels.detach().cpu(), remove_bg=True)

                val_loss += current_loss
                val_accuracy += np.mean(metrics["accuracy"])
                val_precision += np.mean(metrics["precision"])
                val_recall += np.mean(metrics["recall"])
                val_specificity += np.mean(metrics["specificity"])
                val_dice += np.mean(metrics["dice"])
                val_per_class_dice += metrics["dice"]

                num_val_batches += 1

                #pbar.update(1)

            # Write to tensorboard
            overlay_fig, _ = create_overlay_figure(inputs, labels, outputs, images_per_batch=4)

            writer.add_figure("images/val", overlay_fig, epoch)

            writer.add_scalar("validation_metrics/accuracy", val_accuracy/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/precision", val_precision/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/recall", val_recall/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/specificity", val_specificity/num_val_batches, epoch)
            writer.add_scalar("validation_metrics/dice", val_dice/num_val_batches, epoch)

            for i, name in enumerate(net.class_names):
                writer.add_scalar(f"dice/{name}", val_per_class_dice[i]/num_val_batches, epoch)

            writer.add_scalar("loss/validation", val_loss/num_val_batches, epoch)

            writer.add_scalar("loss/train", train_loss/num_batches, epoch)

            writer.add_scalar("learning rate", optimizer.param_groups[0]['lr'], epoch)

            print(f"done! Train/val loss: {round(train_loss/num_batches, 4)}/{round(val_loss/num_val_batches, 4)} ", end="")

        # Save model checkpoints
        torch.save({
            "epoch": epoch,
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
        }, checkpoint_path)

        # Step learning rate scheduler
        scheduler.step(val_dice/num_val_batches)

        elapsed_time = time() - start_time
        print(f"(took {elapsed_time}ms)")

    # Save final model
    torch.save(net.state_dict(), model_path)
    #pbar.close()

    print('Finished Training')