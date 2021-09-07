"""Run reconstruction in a terminal prompt.

Optional arguments can be found in inversefed/options.py

Set as arguments:
- model
- dataset (ImageNet)

- trained_model (pretrained)
- epochs (of the trained model)

- num_images (images to recover)
- target_id (-1 means custom image)

- optim (ours, zhu (dlg))

- save_image (save output)
- deterministic flag
- dryrun flag (run everything for one step for testing)
- name (for the experiment)
"""
import os

selected_gpus = [0] #configure this
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(gpu) for gpu in selected_gpus])


import torch
import torchvision
from torchvision import transforms
from torchvision import models
import torch.nn as nn

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

import inversefed
from inversefed.data.loss import Classification, BCE_Classification

# load modified models
import custom_models

from collections import defaultdict
import datetime
import time

# torch.backends.cudnn.benchmark = inversefed.consts.BENCHMARK
# use_cuda = torch.cuda.is_available()
# device = 'cuda' if use_cuda else 'cpu'

# Parse input arguments
args = inversefed.options().parse_args()

# Parse training strategy (for nn training)
defs = inversefed.training_strategy('conservative')
defs.epochs = args.epochs

# more parameters
num_classes = 1
label_encoding = 'multi' # 'one-hot' or 'multi'
num_channels = 3
change_inchannel = False # change channels of given model from 3 to 1 (greyscale)
random_seed = 207

demo_img_path = 'xray_test.jpg'
img_size = (224,224)
img_label = 1

loss_name = 'CE'
# only one label for now
if label_encoding == 'multi':
    img_label = torch.Tensor([img_label]).float()
    img_label = img_label.view(1,1,)
    loss_name = 'BCE'

restarts = 3
max_iterations = 20000
init = 'custom'

# CheXpert mean and std
xray_mean = 0.5029
xray_std = 0.2899

# partly overwrites bash arguments
set_config = dict(signed=args.signed, # True
              boxed=args.boxed, # True
              cost_fn=args.cost_fn, # cosine sim.
              indices='def',
              weights='equal',
              lr=0.1,
              optim=args.optimizer, # adam, sgd, adamw, lbfgs
              restarts=restarts,
              max_iterations=max_iterations,
              total_variation=args.tv,
              init=init,
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

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

    # mean, std if input channels are modified (greyscale)
    if change_inchannel:
        dm = xray_mean
        ds = xray_std
    else:
    # mean, std if a pretrained model with unmodified input is used
        dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
        ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]

    tt = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    # currently supported models
    if args.model == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=args.trained_model)
        model_seed = None
    elif args.model == 'ResNet18':
        model_seed = None
        if label_encoding == 'multi':
            model = custom_models.ResNet18(out_size=num_classes, pre_trained=args.trained_model)
        else:
            model = torchvision.models.resnet18(pretrained=args.trained_model)

    elif args.model == 'DenseNet121':
        model = models.densenet121(pretrained = args.trained_model)
        model_seed = None
    else:
        exit('Model not supported')

    # change number of input channels from 3 (RGB) to 1 (grey), tested for ResNet18
    # https://discuss.pytorch.org/t/how-to-transfer-the-pretrained-weights-for-a-standard-resnet50-to-a-4-channel/52252
    # https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
    if change_inchannel:
        if type(model).__name__ == 'ResNet18':
            conv1_weight = model.resnet18.conv1.weight.clone()
            model.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                model.resnet18.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights
        else:
            conv1_weight = model.conv1.weight.clone()
            model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            with torch.no_grad():
                model.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True)) # way to keep pretrained weights
        print(model)

    model.to(**setup)
    model.eval()

    # Choose example images from the validation set or from third-party sources
    if args.num_images == 1:
        if args.target_id == -1:  # demo, custom image

            if change_inchannel:

                ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).resize(img_size))/255, **setup)
                print(ground_truth)
                ground_truth = ground_truth.view(1,1,*ground_truth.size())
                ground_truth = ground_truth.sub(xray_mean).div(xray_std) # normalize

                plt.imshow(tp(torch.cat((ground_truth,ground_truth,ground_truth),1)[0].cpu()))
                plt.show()

                img_shape = (1, ground_truth.shape[2], ground_truth.shape[3])

            else: # original RGB image
                # Specify PIL filter for lower pillow versions
                ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).convert('RGB').resize(img_size, Image.BICUBIC)) / 255, **setup)
                # print(ground_truth)
                ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()

                img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

            print(ground_truth)
            print(ground_truth.shape)

            # if not args.label_flip:
            #     labels = torch.as_tensor((img_label,), device=setup['device'])
            #     print(labels)
            # else: # this won't matter here
            #     labels = torch.as_tensor((5,), device=setup['device'])
            if label_encoding == 'multi':
                labels = img_label.to(setup['device'])
            else:
                labels = torch.as_tensor((img_label,), device=setup['device'])
            print(labels)
            target_id = -1

        else: # when using a predefined dataset, not relevant here
            if args.target_id is None:
                target_id = np.random.randint(len(validloader.dataset))
            else:
                target_id = args.target_id
            ground_truth, labels = validloader.dataset[target_id]
            if args.label_flip:
                labels = torch.randint((10,))
            ground_truth, labels = ground_truth.unsqueeze(0).to(**setup), torch.as_tensor((labels,), device=setup['device'])


        # plt.imshow(tp(ground_truth[0].cpu()))
        # ground_truth_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        # plt.imshow(tp(ground_truth_denormalized[0].cpu()))
        # plt.show()
        # print(ground_truth_denormalized)

    # adapt this for multiple images without a predefined dataset
    else:
        ground_truth, labels = [], []
        if args.target_id is None:
            target_id = np.random.randint(len(validloader.dataset))
        else:
            target_id = args.target_id
        while len(labels) < args.num_images:
            img, label = validloader.dataset[target_id]
            target_id += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=setup['device']))
                ground_truth.append(img.to(**setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        if args.label_flip:
            labels = torch.permute(labels)

        img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

    # Run reconstruction
    if args.accumulation == 0: # one epoch, no fed averaging

        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]

        #--------- not of interest here
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
        #----------

        if args.optim == 'ours':
            config = set_config

        else:
            exit("Modify the configurations if you want to change the optimization options")

        # reconstruction process
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images, loss_fn=loss_name)
        output, stats = rec_machine.reconstruct(input_gradient, labels=labels, img_shape=img_shape, dryrun=args.dryrun)


    else:
        pass
        # fed averaging not of interest here

    #     local_gradient_steps = args.accumulation
    #     local_lr = 1e-4
    #     input_parameters = inversefed.reconstruction_algorithms.loss_steps(model, ground_truth, labels,
    #                                                                        lr=local_lr, local_steps=local_gradient_steps)
    #     input_parameters = [p.detach() for p in input_parameters]
    #
    #     # Run reconstruction in different precision?
    #     if args.dtype != 'float':
    #         if args.dtype in ['double', 'float64']:
    #             setup['dtype'] = torch.double
    #         elif args.dtype in ['half', 'float16']:
    #             setup['dtype'] = torch.half
    #         else:
    #             raise ValueError(f'Unknown data type argument {args.dtype}.')
    #         print(f'Model and input parameter moved to {args.dtype}-precision.')
    #         ground_truth = ground_truth.to(**setup)
    #         dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
    #         ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    #         input_parameters = [g.to(**setup) for g in input_parameters]
    #         model.to(**setup)
    #         model.eval()
    #
    #     config = dict(signed=args.signed,
    #                   boxed=args.boxed,
    #                   cost_fn=args.cost_fn,
    #                   indices=args.indices,
    #                   weights=args.weights,
    #                   lr=1,
    #                   optim=args.optimizer,
    #                   restarts=args.restarts,
    #                   max_iterations=24_000,
    #                   total_variation=args.tv,
    #                   init=args.init,
    #                   filter='none',
    #                   lr_decay=True,
    #                   scoring_choice=args.scoring_choice)
    #
    #     rec_machine = inversefed.FedAvgReconstructor(model, (dm, ds), local_gradient_steps, local_lr, config,
    #                                                  num_images=args.num_images, use_updates=True)
    #     output, stats = rec_machine.reconstruct(input_parameters, labels, img_shape=img_shape, dryrun=args.dryrun)
    #
    #


    # Compute stats
    factor=1/ds # for psnr

    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    # feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    feat_mse = np.nan # placeholder so no errors occur
    test_psnr = inversefed.metrics.psnr(output.detach(), ground_truth, factor=factor)

    # Save the best image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        rec_filename = f'{args.name}_rec_img_idx{stats["best_exp"]}.png'
        torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

        gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        gt_filename = f'{args.name}_gt_img.png'
        torchvision.utils.save_image(gt_denormalized, os.path.join(args.image_path, gt_filename))
    else:
        rec_filename = None
        gt_filename = None

    # save stats
    for trial in rec_machine.exp_stats:
        mses = [((rec_img - ground_truth).pow(2).mean().item()) for rec_img in trial['history']]
        psnrs = [(inversefed.metrics.psnr(rec_img, ground_truth, factor=factor)) for rec_img in trial['history']]
        all_metrics = [trial['idx'], trial['rec_loss'], mses, psnrs]
        with open(f'trial_histories/{args.name}_{trial["name"]}.csv', 'w') as f:
            header = ['iteration', 'loss', 'mse', 'psnr']
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(zip(*all_metrics))

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} |")

    # save parameters
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
                                   tv=args.tv,

                                   rec_loss=stats["opt"],
                                   best_idx=stats["best_exp"],
                                   psnr=test_psnr,
                                   test_mse=test_mse,
                                   feat_mse=feat_mse,

                                   target_id=target_id,
                                   seed=model_seed,
                                   timing=str(datetime.timedelta(seconds=time.time() - start_time)),
                                   dtype=setup['dtype'],
                                   epochs=defs.epochs,
                                   val_acc=None,
                                   rec_img=rec_filename,
                                   gt_img=gt_filename
                                   )


    # Print final timestamp
    print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
    print('---------------------------------------------------')
    print(f'Finished computations with time: {str(datetime.timedelta(seconds=time.time() - start_time))}')
    print('-------------Job finished.-------------------------')
