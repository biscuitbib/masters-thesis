import torch
from tqdm import tqdm, trange

def predict_volume(net, image):
    """
    make prediction on image volume using the three standard axes
    """
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    h, w, d = image.shape
    
    with torch.no_grad():
        prediction_image = torch.zeros((net.n_classes, h, w, d), dtype=image.dtype)
        for i in trange(h):
            image_slice = image[i, :, :]
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)

            prediction_image[:, i, :, :] += prediction.squeeze().detach().cpu()

        for j in trange(w):
            image_slice = image[:, j, :]
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)

            prediction_image[:, :, j, :] += prediction.squeeze().detach().cpu()

        for k in trange(d):
            image_slice = image[:, :, k]
            image_slice = image_slice.unsqueeze(0).unsqueeze(0).to(device)
            prediction = net(image_slice)

            prediction_image[:, :, :, k] += prediction.squeeze().detach().cpu()

        prediction_volume = torch.argmax(prediction_image, dim=0)
        
    return prediction_volume
    
    