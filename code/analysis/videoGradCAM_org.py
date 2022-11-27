"""
 * @author Gautam R Gare
 * @email gautam.r.gare@gmail.com
 * @create date 2021-11-02 17:53:29
 * @modify date 2021-11-03 21:22:11
 * @desc [description]
 """
 
import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms

import visualizeAnnotation

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, model_name, feature_module, target_layers):
        self.model = model
        self.model_name = model_name
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        
        for name_m, module_m in self.model._modules.items():
            if name_m == "base_model":
                for name, module in module_m._modules.items():
                    if module == self.feature_module:
                        target_activations, x = self.feature_extractor(x)
                    elif "avgpool" in name.lower():
                        x = module(x)
                        x = x.view(x.size(0),-1)
                    else:
                        x = module(x)
            elif "consensus" in name_m.lower(): #TODO - GRG: Check if we need consensus module
                x = x.unsqueeze(0)
                x = module_m(x)
                x = x.squeeze(1)
                # pred = x.unsqueeze(0)
                # pred = module_m(pred)
                # pred = pred.squeeze(1)
            else:
                x = module_m(x)

        

        return target_activations, x
        # return target_activations, x, pred

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    #Convert RGB to BGR for numpy saving
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


class GradCam:
    # def __init__(self, model, feature_module, target_layer_names, use_cuda):
    def __init__(self, model, model_name, feature_module, target_layer_names, device, contrastive_model = False):
        self.model = model
        self.model_name = model_name
        self.feature_module = feature_module
        self.model.eval()
        # self.cuda = use_cuda
        # if self.cuda:
        #     self.model = model.cuda()
        self.device = device
        self.model = self.model.to(device)

        self.extractor = ModelOutputs(self.model, self.model_name, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
       
        input_img = input_img.to(self.device)

        features, output = self.extractor(input_img)
        # features, output, pred = self.extractor(input_img)

        # pred_category = np.argmax(pred.cpu().data.numpy())

        #TODO - GRG : #Note-GRG: The 'target_category' make it as method variable to set!!
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())
            # target_category = pred_category

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = np.zeros(output.size(), dtype=np.float32)
        one_hot[:, target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        one_hot = one_hot.to(self.device)
        
        one_hot = torch.sum(one_hot * output)
        # one_hot = torch.sum(one_hot * output, dim=1)
        
        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        # target = target.cpu().data.numpy()[0, :]
        target = target.cpu().data.numpy()

        # weights = np.mean(grads_val, axis=(2, 3))[0, :]
        weights = np.mean(grads_val, axis=(2, 3))
        # cam = np.zeros(target.shape[1:], dtype=np.float32)
        cam_f = np.zeros((target.shape[0], target.shape[-2], target.shape[-1]), dtype=np.float32)

        for f, f_w in enumerate(weights):
            for i, w in enumerate(f_w):
                cam_f[f] += w * target[f, i, :, :]

        input_img = input_img.squeeze().cpu().data.numpy()
        cam_clip = []
        grayscale_cam_clip = []
        for cam, img in zip(cam_f, input_img):
            cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, input_img.shape[-2:])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)

            grayscale_cam_clip.append(cam)

            cam = show_cam_on_image(img[:, :, np.newaxis], cam)

            cam_clip.append(cam)

        cam_clip = np.array(cam_clip)
        grayscale_cam_clip = np.array(grayscale_cam_clip)

        return cam_clip, grayscale_cam_clip, target_category


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input



import copy 

class GuidedBackpropReLUModel:
    def __init__(self, model, device):
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.device = device
       
        self.model = self.model.to(device)

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, grayscale_cam, target_category=None):
        
        input_img = input_img.to(self.device)

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        
        one_hot = one_hot.to(self.device)

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        # output = output[0, :, :, :]

        output = output.squeeze()[:,:,:,np.newaxis]

        cam_gb_clip = []
        gb_clip = []
        for gb, cam in zip(output, grayscale_cam):
            cam_mask = cv2.merge([cam, cam, cam])
            cam_gb = deprocess_image(cam_mask*gb)
            gb = deprocess_image(gb)

            cam_gb_clip.append(cam_gb)
            gb_clip.append(gb)
        
        cam_gb_clip = np.array(cam_gb_clip)
        gb_clip = np.array(gb_clip)

        return gb_clip, cam_gb_clip

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)




class VideoGradCAM(object):

    def __init__(self, model, feature_module, target_layer_names, device, target_category = None):
        
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        self.target_category = target_category

        self.grad_cam = GradCam(model, '', feature_module, target_layer_names, device)

        self.gb_model = GuidedBackpropReLUModel(model, device)


    def getCAM(self, input_img):

        cam, grayscale_cam, prediction = self.grad_cam(input_img, self.target_category)

        # grayscale_cam = cv2.resize(grayscale_cam, (img.shape[-1], img.shape[-2]))

        # cam = show_cam_on_image(img.squeeze().unsqueeze(2), grayscale_cam)
        
        gb, cam_gb = self.gb_model(input_img, grayscale_cam, self.target_category)
        # gb = gb.transpose((1, 2, 0))
        

        return cam, gb, cam_gb, prediction
            


def visualizeVideoGradCamActiviationForDataset(self, model, feature_module, target_layer_names, dataloader, filename_array, grey_array, label_array, device):

    if self.plotVideoGradCam:
        utils.removeDir(constants.video_gradcam_path)
        utils.createDir(constants.video_gradcam_path)
    else:
        return
    
    class_names = dataloader.dataset.label_names

    videoGradCAM = VideoGradCAM(model, feature_module = feature_module, target_layer_names = target_layer_names, device = device)

    # input_clips = []
    # target_activity = []
    # pred_activity = []
    for i in range(len(filename_array)):
        filename = filename_array[i]
        clip = grey_array[i]
        # rf = rf_array[i]
        label = label_array[i]

        cam, gb, cam_gb, prediction = videoGradCAM.getCAM(clip)

        if self.args.biomarker_labels:
            target = ";".join([class_names[i] for i, a in enumerate(label) if a == 1])
            pred = class_names[prediction]
        else:
            target = class_names[label]
            pred = class_names[prediction]

        visualizeAnnotation.saveGradCAMPredAsGIF(clip, cam, gb, cam_gb, target, pred, constants.analysis_dir_path, class_names, idx = i, prefix = filename, biomarker_labels = self.args.biomarker_labels)



def visualizeVideoGradCamActiviation(self, model, feature_module, target_layer_names, 
    class_names, filename, clip, label, device):

    videoGradCAM = VideoGradCAM(model, feature_module = feature_module, target_layer_names = target_layer_names, device = device)


    cam, gb, cam_gb, prediction = videoGradCAM.getCAM(clip)

    if self.args.biomarker_labels:
        target = ";".join([class_names[i] for i, a in enumerate(label) if a == 1])
        pred = class_names[prediction]
    else:
        target = class_names[label]
        pred = class_names[prediction]

    visualizeAnnotation.saveGradCAMPredAsGIF(clip, cam, gb, cam_gb, target, pred, constants.analysis_dir_path, class_names, prefix = filename)



if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    model = models.resnet50(pretrained=True)
    model_name='resnet50'

    # grad_cam = GradCam(model=model, feature_module=model.layer4, \
    #                    target_layer_names=["2"], use_cuda=args.use_cuda)


    device = 'cpu'
    if args.use_cuda:
        device = 'cuda'
    grad_cam = GradCam(model=model, model_name=model_name, feature_module=model.layer4, \
                       target_layer_names=["2"], device=device)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    input_img = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None
    grayscale_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    gb = gb_model(input_img, grayscale_cam, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite("cam.jpg", cam)
    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)

