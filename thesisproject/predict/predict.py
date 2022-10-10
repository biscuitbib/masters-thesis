import os
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from thesisproject.utils import grayscale_to_rgb, segmentation_to_rgb

class Predict:
    def __init__(self, net, batch_size=8, show_progress=True):
        self.net = net
        self.show_progress = show_progress
        self.pbar = None
        self.batch_size = batch_size
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def _predict_view(self, image, view=""):
        self.net.eval()
        with torch.no_grad():
            tmp_vol = None
            i = 0
            for image_batch in torch.split(image, self.batch_size, dim=0):
                image_batch = image_batch.unsqueeze(1).to(self._device)
                    
                prediction = self.net(image_batch).squeeze(1).detach().cpu()
                
                """
                ########## SAVE PER VIEW IMAGES
                input_imgs = image_batch.detach().cpu()
                input_imgs /= torch.max(input_imgs)
                input_imgs = grayscale_to_rgb(input_imgs)

                b, h, w, c = input_imgs.shape

                pred_imgs = segmentation_to_rgb(prediction.detach().cpu())
                pred_overlay = (input_imgs / 2) + (pred_imgs / 2)
                pred_overlay = torch.cat([pred_overlay[i, ...] for i in range(pred_overlay.shape[0])], dim=1).numpy()
                
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.imshow(pred_overlay)
                ax.set_title(f"Prediction batch along {view} view")
                ax.set_xticks([])
                ax.set_yticks([])
                plt.tight_layout()
                fig.savefig(f"predictions/per_slice/{view}_{i}.jpg")
                
                plt.close()
                i += 1
                ###################
                """

                if tmp_vol is None:
                    tmp_vol = prediction
                else:
                    tmp_vol = torch.cat([tmp_vol, prediction], dim=0)

                if self.show_progress:
                    self.pbar.update(image_batch.shape[0])
                    
            tmp_vol = tmp_vol.permute(1, 0, 2, 3)
            return F.softmax(tmp_vol, dim=0)
        
    def predict_volume(self, image, class_index=True):
        if self.show_progress:
            self.pbar = tqdm(total=(np.sum(image.shape)), unit="slice")

        prediction_image = torch.zeros((self.net.n_classes, *image.shape), dtype=image.dtype)

        if self.show_progress:
            self.pbar.set_description("First view")
        prediction_image += self._predict_view(image, view="axial")

        if self.show_progress:
            self.pbar.set_description("Second view")
        rot_img = image.permute(1, 0, 2)
        prediction_image += self._predict_view(rot_img, view="coronal").permute(0, 2, 1, 3)
        
        if self.show_progress:
            self.pbar.set_description("Third view")
        rot_img = image.permute(2, 0, 1)
        prediction_image += self._predict_view(rot_img, view="saggittal").permute(0, 2, 3, 1)

        if self.show_progress:
            self.pbar.close()
            self.pbar = None
            
        if class_index:
            return torch.argmax(prediction_image, dim=0)
        else:
            return prediction_image / 3.
    

    def __call__(self, image, class_index=True):
        return self.predict_volume(image, class_index=class_index)
    
    