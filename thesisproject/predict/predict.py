import torch
import torch.nn.functional as F
from tqdm import tqdm, trange

def predict_volume(net, image):
    """
    make prediction on image volume using the three standard axes
    """
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    h, w, d = image.shape
    
    #pbar = tqdm(total=(h+w+d))
    
    with torch.no_grad():
        prediction_image = torch.zeros((net.n_classes, h, w, d), dtype=image.dtype)
        #pbar.set_description("First dim")
        for i in range(h):
            image_slice = image[i, :, :]
            if torch.sum(image_slice) == 0:
                continue
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)
            pred_softmax = F.softmax(prediction.squeeze().detach().cpu(), dim=0)

            prediction_image[:, i, :, :] += pred_softmax
            #pbar.update(1)

        #pbar.set_description("Second dim")
        for j in range(w):
            image_slice = image[:, j, :]
            if torch.sum(image_slice) == 0:
                continue
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)
            pred_softmax = F.softmax(prediction.squeeze().detach().cpu(), dim=0)

            prediction_image[:, :, j, :] += pred_softmax
            #pbar.update(1)

        #pbar.set_description("Third dim")
        for k in range(d):
            image_slice = image[:, :, k]
            if torch.sum(image_slice) == 0:
                continue
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)
            pred_softmax = F.softmax(prediction.squeeze().detach().cpu(), dim=0)

            prediction_image[:, :, :, k] += pred_softmax
            #pbar.update(1)
            
        
        #pbar.close()
        prediction_volume = torch.argmax(prediction_image, dim=0)
        
    return prediction_volume
    
    