"""Run reconstruction in a terminal prompt.
Adapted from https://github.com/JonasGeiping/invertinggradients/blob/master/reconstruct_image.py.

Optional arguments can be found in inversefed/options.py

When run in terminal, set the following options:
--model (model type)
--name (for file saving)
--dataset (relevant if using pretrained model)
--optimizer (attack optimizer, usually adam)
--num_images (number of input images)
--trained_model (if using a pretrained model)
--save_images (saving results)
--deterministic (necessary for reproducibility)
--dryrun (for a test run)

Script usage example:
python reconstruct_cxr.py --model ResNet50 --name resnet50 --dataset ImageNet --optimizer adam --num_images 1 --trained_model --save_image --deterministic

"""
import os

selected_gpus = [7] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])

import torch
import torchvision
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from torch import optim

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv
import pandas as pd

import inversefed
from inversefed.data.loss import Classification, BCE_Classification

# load modified models
import custom_models
from custom_models import weights_init, freeze_batchnorm
from module_modification import convert_batchnorm_modules

from collections import defaultdict
import datetime
import time
from copy import deepcopy

# Parse input arguments
args = inversefed.options().parse_args()

# ---- SET MORE PARAMETERS HERE ---- #
random_seed = 207

# LABELS
# 1 if binary classification
num_classes = 1
# 'one-hot' or 'multi', for one hot encoding or multi label classification
label_encoding = 'multi'

# IMAGE DATA
# leave this if using a pretrained model
num_channels = 3
# change channels of given model from 3 to 1 (greyscale)
change_inchannel = True
# nothing to do here
if change_inchannel:
    colour_input = 'L'
else:
    colour_input = 'RGB'
# other image parameters
img_size = (224,224)
# normalization mode; 'imgnet' or 'xray'; values for normalization for the case of 1-channel model
# otherwise normalization values for dataset specified in script flag are used
norm = 'imgnet'

# HOW TO COMPUTE ORIGINAL GRADIENTS
# if True, recovering from weights instead of gradients
from_weights = True
# set model lr if recovering from weights
model_lr = 0.01

# MODEL SETTINGS
# set overall model in train mode, or in eval mode if false
train_mode = True
# partial layer freezing, i.e. batch norm layers
set_batchnorm_freeze = True
# whether to read in model parameters from checkpoints
read_init_model = True
read_trained_model = True
# set model checkpoint paths here
if read_init_model:
    init_model_path = '../resnet_bn_freeze/global_0rounds.pth.tar'
if read_trained_model:
    trained_model_path = '../resnet_bn_freeze/round1_client19/1-epoch_FL.pth.tar'
    assert from_weights == True, "Can only infer gradients from model weights"

if read_trained_model:
    # adapt this to model's data for validation, PSNR computation etc.
    data_indices = [0] # check this from FL
    img_label = [0]
    client_file_path = '~/netstore/data_files/combined_files/client19/'
    img_data_path = '/mnt/dsets/'

    client_file = pd.read_csv(client_file_path+'client_train.csv')
    demo_img_path = [img_data_path + client_file['Path'][i] for i in data_indices]
    print("Image paths: ", demo_img_path)
    if args.num_images == 1:
        demo_img_path = demo_img_path[0]
        img_label = img_label[0]
    # demo_img_path = '/mnt/dsets/mendeley_xray/train/PNEUMONIA/VIRUS-4615614-0010.jpeg'
    # img_label = 0

else: # demo mode
    if args.num_images == 1:
        demo_img_path = 'xray_test.jpg'
        img_label = 1
    else:
        demo_folder = 'demo_images/'
        # demo_img_path = [demo_folder+'NORMAL-1455093-0001.jpeg', demo_folder+'VIRUS-6709337-0001.jpeg']
        # img_label = [1,1]

        demo_img_path =[demo_folder+'NORMAL-1455093-0001.jpeg', demo_folder+'VIRUS-6709337-0001.jpeg',
                            demo_folder+'BACTERIA-3000214-0003.jpeg', demo_folder+'NORMAL-9427315-0001.jpeg']
        img_label = [[0,0],[0,1],[1,0],[1,1]]
        # img_label = [[1], [0], [0], [1]]

        assert len(demo_img_path) == args.num_images, "Specified number of images must match image names"
        assert len(img_label) == args.num_images, "Incorrect number of labels provided"

# ATTACK SETTINGS
restarts = 1
max_iterations = 5000
# dummy image initialization mode
# 'randn', 'rand', 'zeros', 'xray', 'mean_xray'
init = 'randn'
# total variation value for cosine similarity loss
# 1e-4 for smaller networks, 1e-1 for larger
tv = 1e-4
# if True, optimize on signed gradients
set_signed = True
# learning rate for attack optimizer
attack_lr = 0.1

# MODEL LOSS FUNCTION, takes care of itself
loss_name = 'CE'
if label_encoding == 'multi':
    img_label = torch.Tensor([img_label]).float()
    img_label = img_label.view(args.num_images,num_classes)
    loss_name = 'BCE'

# IMAGE DATA MEAN and STD for normalization
if norm == 'xray':
    dm = 0.5029
    ds = 0.2899
elif norm == 'imgnet':
    dm = np.mean([0.485, 0.456, 0.406])
    ds = np.mean([0.229, 0.224, 0.225])
else:
    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]

# parameters that always stay the same for the experiments are hardcoded
# or taken from bash arguments
# partly overwrites bash arguments
set_config = dict(signed=set_signed,
              boxed=args.boxed, # True
              cost_fn=args.cost_fn, # cosine sim.
              indices='def',
              weights='equal',
              lr=attack_lr,
              optim=args.optimizer, # adam, sgd, adamw, lbfgs
              restarts=restarts,
              max_iterations=max_iterations,
              total_variation=tv,
              init=init,
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

# ---- END PARAMETER SETTING ---- #

# not entirely reproducible... only on GPU
if args.deterministic:
    inversefed.utils.set_deterministic()
    inversefed.utils.set_random_seed(random_seed)


if __name__ == "__main__":

    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    if label_encoding == 'one-hot':
        loss_fn = Classification() # this is cross entropy, see https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/data/loss.py
    elif label_encoding == 'multi':
        loss_fn = BCE_Classification()

    tp = transforms.Compose([transforms.ToPILImage()])

    # currently supported models
    if args.model == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=args.trained_model)
        model_seed = None

    elif args.model == 'ResNet50':
        model_seed = None
        if label_encoding == 'multi':
            model = custom_models.ResNet50(out_size=num_classes, colour_input=colour_input, pre_trained=args.trained_model)
        else:
            model = torchvision.models.resnet50(pretrained=args.trained_model)

    elif args.model == 'ResNet18':
        model_seed = None
        if label_encoding == 'multi':
            model = custom_models.ResNet18(out_size=num_classes, colour_input=colour_input, pre_trained=args.trained_model)
        else:
            model = torchvision.models.resnet18(pretrained=args.trained_model)

    elif args.model == 'DenseNet121':
        # model = models.densenet121(pretrained = args.trained_model)
        model = custom_models.DenseNet121(out_size=num_classes, colour_input=colour_input, pre_trained=args.trained_model)
        model_seed = None
    else:
        exit('Model not supported')
    # print(model)

    # read initial model checkpoint
    if read_init_model:
        init_model_checkpoint = torch.load(init_model_path)
        if 'state_dict' in init_model_checkpoint:
            model.load_state_dict(init_model_checkpoint['state_dict'])
        else:
            model.load_state_dict(init_model_checkpoint)

    # apply model modifications
    model.to(**setup)

    if train_mode:
        model.train()
        set_eval=False
    else:
        model.eval()
        set_eval=True

    if set_batchnorm_freeze:
        freeze_batchnorm(model)

    # read in images, prepare labels
    if args.num_images == 1:

        # greyscale processing
        if change_inchannel:

            # same preprocessing as in FL
            # train_transformSequence = transforms.Compose([transforms.Resize(img_size),
            #                                         transforms.ToTensor(),
            #                                         transforms.Normalize(dm, ds)
            #                                         ])
            #
            # ground_truth = Image.open(demo_img_path).convert(colour_input) # RGB or L for greyscale
            # ground_truth = train_transformSequence(ground_truth)
            # ground_truth = ground_truth.view(1,*ground_truth.size())

            ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).resize(img_size))/255, **setup)
            ground_truth = ground_truth.view(1,1,*ground_truth.size())
            ground_truth = ground_truth.sub(dm).div(ds) # normalize

            plt.imshow(tp(torch.cat((ground_truth,ground_truth,ground_truth),1)[0].cpu()))
            plt.show()

            img_shape = (1, ground_truth.shape[2], ground_truth.shape[3])

        # original RGB image processing
        else:
            # Specify PIL filter for lower pillow versions
            ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).convert('RGB').resize(img_size, Image.BICUBIC)) / 255, **setup)
            # print(ground_truth)
            ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()

            img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        print("GT: ", ground_truth)
        print("GT image shape: ,", ground_truth.shape)

        if label_encoding == 'multi':
            labels = img_label.to(setup['device'])
        else:
            labels = torch.as_tensor((img_label,), device=setup['device'])
        print("Label shape :,", labels.shape)
        print("Labels: ", labels)

    # same for multiple images
    else:

        if change_inchannel:

            ground_truth = []
            for img_path in demo_img_path:
                img = torch.as_tensor(np.array(Image.open(img_path).resize(img_size))/255, **setup)
                img = img.view(1,*img.size())
                img = img.sub(dm).div(ds) # normalize
                ground_truth.append(img.to(**setup))
            ground_truth = torch.stack(ground_truth)
            img_shape = (1, ground_truth.shape[2], ground_truth.shape[3])

        else:
            ground_truth= []
            for img_path in demo_img_path:
                img = torch.as_tensor(np.array(Image.open(img_path).convert('RGB').resize(img_size, Image.BICUBIC)) / 255, **setup)
                img = img.permute(2, 0, 1).sub(dm).div(ds)
                ground_truth.append(img.to(**setup))
            ground_truth = torch.stack(ground_truth)
            img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

        plt.imshow(tp(ground_truth[0].cpu()))
        plt.show()
        print("GT: ", ground_truth)
        print("GT image shape: ,", ground_truth.shape)
        labels = torch.as_tensor(img_label, device=setup['device'])
        print("Label shape :,", labels.shape)
        print("Labels: ", labels)

    # Compute original gradients
    if from_weights:

        placeholder_model = deepcopy(model) # copy model so optim step will not be recorded on original
        model.zero_grad()
        placeholder_model.zero_grad()
        initial_parameters = deepcopy(model.state_dict()) # we care only about parameters

        # take optimization step if no trained model is available
        if not read_trained_model:

            # optimization step on placeholder model
            model_optim = optim.SGD(placeholder_model.parameters(), lr = model_lr)
            target_loss, _, _ = loss_fn(placeholder_model(ground_truth), labels)
            target_loss.backward()
            model_optim.step()

            # approximately compute back gradients
            new_parameters = placeholder_model.state_dict()
            with torch.no_grad():
                check_params = model.parameters()
                input_gradient = []
                for key in new_parameters:
                    if key.endswith('weight') or key.endswith('bias'):
                        if(next(check_params).requires_grad): # only compute gradients for layers that are not frozen
                            cur_grad = -(new_parameters[key] - initial_parameters[key])/model_lr
                            # replace weird negative zeros with proper zero values, to be sure
                            cur_grad = torch.where(cur_grad == -0.0000e+00, torch.tensor(0.).to(**setup), cur_grad)
                            input_gradient.append(cur_grad.detach())
            # print(input_gradient[41])
            print("Length of original gradient: ", len(input_gradient))

        else:

            # load trained model
            trained_model_checkpoint = torch.load(trained_model_path)
            if 'state_dict' in trained_model_checkpoint:
                placeholder_model.load_state_dict(trained_model_checkpoint['state_dict'])
            else:
                placeholder_model.load_state_dict(trained_model_checkpoint)

            new_parameters = placeholder_model.state_dict()

            # compute gradients from loaded model
            with torch.no_grad():
                check_params = model.parameters()
                input_gradient = []
                for key in new_parameters:
                    if key.endswith('weight') or key.endswith('bias'):
                        if(next(check_params).requires_grad): # only compute gradients for layers that are not frozen
                            cur_grad = -(new_parameters[key] - initial_parameters[key])/model_lr
                            # replace weird negative zeros with proper zero values, to be sure
                            cur_grad = torch.where(cur_grad == -0.0000e+00, torch.tensor(0.).to(**setup), cur_grad)
                            input_gradient.append(cur_grad.detach())
            # print(input_gradient[41])

    else: # compute gradients directly

        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        # https://discuss.pytorch.org/t/how-the-pytorch-freeze-network-in-some-layers-only-the-rest-of-the-training/7088/9
        input_gradient = torch.autograd.grad(target_loss, filter(lambda p: p.requires_grad, model.parameters()))
        input_gradient = [grad.detach() for grad in input_gradient]
        print("Length of original gradient: ", len(input_gradient))
        # print(input_gradient[41])

        # --- not of interest here --- #
        # full_norm = torch.stack([g.norm() for g in input_gradient]).mean()
        # print(f'Full gradient norm is {full_norm:e}.')

        # Run reconstruction in different precision?
        if args.dtype != 'float':
            if args.dtype in ['double', 'float64']:
                setup['dtype'] = torch.double
            elif args.dtype in ['half', 'float16']:
                setup['dtype'] = torch.half
            else:
                raise ValueError(f'Unknown data type argument {args.dtype}.')
            print(f'Model and input parameter moved to {args.dtype}-precision.')

            ground_truth = ground_truth.to(**setup)
            input_gradient = [g.to(**setup) for g in input_gradient]
            model.to(**setup)
            model.eval()
        # --- --- #

    # --- also ignore this --- #
    if args.optim == 'ours':
        config = set_config

    else:
        exit("Modify the configurations if you want to change the optimization options")

    # --- end of ignorance --- #

    # Reconstruct data!
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, loss_fn=loss_name)
    output, stats = rec_machine.reconstruct(input_gradient, labels=labels, img_shape=img_shape, dryrun=args.dryrun, set_eval=set_eval)


    # Compute stats
    factor=1/ds # for PSNR computation

    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    # feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    feat_mse = np.nan # placeholder so no errors occur
    test_psnr = inversefed.metrics.psnr(output.detach(), ground_truth, factor=factor)

    # Save the best reconstructed image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        for img_idx in range(args.num_images):
            rec_filename = f'{args.name}_rec_img_exp{stats["best_exp"]}_idx{img_idx}.png'
            torchvision.utils.save_image(output_denormalized[img_idx], os.path.join(args.image_path, rec_filename))
            gt_filename = f'{args.name}_gt_img_idx{img_idx}.png'
            torchvision.utils.save_image(gt_denormalized[img_idx], os.path.join(args.image_path, gt_filename))
    else:
        rec_filename = None
        gt_filename = None

    # Save stats
    for trial in rec_machine.exp_stats:
        all_mses, all_psnrs = [], []
        for img_hist in trial['history']:
            mses = [((rec_img - gt_img).pow(2).mean().item()) for rec_img, gt_img in zip(img_hist, ground_truth)]
            psnrs = [(inversefed.metrics.psnr(rec_img.unsqueeze(0), gt_img.unsqueeze(0), factor=factor)) for rec_img, gt_img in zip(img_hist, ground_truth)]
            all_mses.append(mses)
            all_psnrs.append(psnrs)
        all_metrics = [trial['idx'], trial['rec_loss'], all_mses, all_psnrs]
        with open(f'trial_histories/{args.name}_{trial["name"]}.csv', 'w') as f:
            header = ['iteration', 'loss', 'mse', 'psnr']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*all_metrics))

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} |")

    # Save parameters in table
    # some values are not recorded (e.g., feat_mse, val_acc)
    inversefed.utils.save_to_table(args.table_path, name=f'exp_{args.name}', dryrun=args.dryrun,

                                   model=args.model,
                                   dataset=args.dataset,
                                   trained=args.trained_model,
                                   accumulation=args.accumulation,
                                   restarts=config['restarts'],
                                   OPTIM=args.optim,
                                   cost_fn=args.cost_fn,
                                   indices=args.indices,
                                   weights=args.weights,
                                   scoring=args.scoring_choice,
                                   init=config['init'],
                                   tv=tv,

                                   rec_loss=stats["opt"],
                                   best_idx=stats["best_exp"],
                                   psnr=test_psnr,
                                   test_mse=test_mse,
                                   feat_mse=feat_mse,

                                   target_id=-1,
                                   seed=model_seed,
                                   timing=str(datetime.timedelta(seconds=time.time() - start_time)),
                                   dtype=setup['dtype'],
                                   epochs=args.epochs,
                                   val_acc=None,
                                   rec_img=rec_filename,
                                   gt_img=gt_filename
                                   )


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
