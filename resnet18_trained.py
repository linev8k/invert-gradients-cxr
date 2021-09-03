import os
selected_gpus = [0] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import torch
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from torchvision import transforms

trained_model = True
arch = 'ResNet18'
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

import inversefed
from inversefed.data.loss import Classification

setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative')

img_size = (224,224)
tt = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
tp = transforms.Compose([transforms.ToPILImage()])

# loss_fn, trainloader, validloader =  inversefed.construct_dataloaders('ImageNet', defs,
                                                                      # data_path='/data/imagenet')

model = torchvision.models.resnet18(pretrained=trained_model)
model.to(**setup)
model.eval()
loss_fn = Classification()

# dm = torch.as_tensor([0.485, 0.456, 0.406], **setup)[:, None, None]
# ds = torch.as_tensor([0.229, 0.224, 0.225], **setup)[:, None, None]
dm = torch.as_tensor([0,0,0], **setup)[:, None, None]
ds = torch.as_tensor([1,1,1], **setup)[:, None, None]

def plot(tensor):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    if tensor.shape[0] == 1:
        # return plt.imshow(tensor[0].permute(1, 2, 0).cpu())
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
        plt.show()
    else:
        fig, axes = plt.subplots(1, tensor.shape[0], figsize=(12, tensor.shape[0]*12))
        for i, im in enumerate(tensor):
            axes[i].imshow(im.permute(1, 2, 0).cpu())
        plt.show()

tmp_datum = Image.open('mandalorian.jpeg').convert('RGB')
tmp_datum = tt(tmp_datum).to(device)
img = tmp_datum.view(1, *tmp_datum.size())
tmp_label = torch.Tensor([1]).long().to(device)
labels = tmp_label.view(1, )

# img, label = validloader.dataset[idx]
# labels = torch.as_tensor((label,), device=setup['device'])
ground_truth = img[0].to(**setup).unsqueeze(0)
# ground_truth = img.to(**setup)
plt.imshow(tp(ground_truth[0][0].cpu()))
plt.show()
# plot(ground_truth)
# print([trainloader.dataset.classes[l] for l in labels])

ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
plt.imshow(tp(ground_truth_denormalized[0][0].cpu()))
plt.show()
# torchvision.utils.save_image(ground_truth_denormalized, f'{idx}_{arch}_ImageNet_input.png')


model.zero_grad()
target_loss, _, _ = loss_fn(model(ground_truth), labels)
input_gradient = torch.autograd.grad(target_loss, model.parameters())
input_gradient = [grad.detach() for grad in input_gradient]
full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
print(f'Full gradient norm is {full_norm:e}.')


config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=1_000,
              total_variation=1e-1,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=(3, 224, 224))

test_mse = (output.detach() - ground_truth).pow(2).mean()
feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
test_psnr = inversefed.metrics.psnr(output, ground_truth)

plot(output)
plt.title(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");

# data = inversefed.metrics.activation_errors(model, output, ground_truth)
