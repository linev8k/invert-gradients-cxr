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
from inversefed.data.loss import Classification

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
num_classes = 2
num_channels = 3
random_seed = 207

demo_img_path = 'xray_test.jpg'
img_size = (224,224)
img_label = 1
greyscale=False #leave this!
read_grey = False
xray_mean = 0.5
xray_std = 0.3


change_inchannel = True # change channels of given model from 3 to 1 (greyscale)


set_config = dict(signed=args.signed,
              boxed=args.boxed,
              cost_fn=args.cost_fn,
              indices='def',
              weights='equal',
              lr=0.1,
              optim=args.optimizer,
              restarts=args.restarts,
              max_iterations=1000,
              total_variation=args.tv,
              init='custom',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

# not reproducible...
if args.deterministic:
    inversefed.utils.set_deterministic()
    inversefed.utils.set_random_seed(random_seed)


if __name__ == "__main__":

    # Choose GPU device and print status information:
    setup = inversefed.utils.system_startup(args)
    start_time = time.time()

    # Get data:
    # loss_fn, trainloader, validloader = inversefed.construct_dataloaders(args.dataset, defs, data_path=args.data_path)

    loss_fn = Classification() # this is cross entropy, see https://github.com/JonasGeiping/invertinggradients/blob/master/inversefed/data/loss.py

    dm = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(getattr(inversefed.consts, f'{args.dataset.lower()}_std'), **setup)[:, None, None]

    if change_inchannel:
        dm = xray_mean
        ds = xray_std

    tt = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])


    # if args.dataset == 'ImageNet':
    if args.model == 'ResNet152':
        model = torchvision.models.resnet152(pretrained=args.trained_model)
        model_seed = None
    elif args.model == 'ResNet18':
        model = torchvision.models.resnet18(pretrained=args.trained_model)
        model_seed = None
    elif args.model == 'DenseNet121':
        model = models.densenet121(pretrained = args.trained_model)
        model_seed = None
    else: # substitute this by other model that I want to use
        model, model_seed = inversefed.construct_model(args.model, num_classes=num_classes, num_channels=num_channels)
        print('Model seed: ', model_seed)

    # print(model)
    if change_inchannel: # for ResNet18!
        # https://discuss.pytorch.org/t/how-to-transfer-the-pretrained-weights-for-a-standard-resnet50-to-a-4-channel/52252
        conv1_weight = model.conv1.weight.clone()
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(conv1_weight.sum(dim=1,keepdim=True))
        # print(model)

        # https://stackoverflow.com/questions/51995977/how-can-i-use-a-pre-trained-neural-network-with-grayscale-images
        # cur_state_dict = model.state_dict()
        # conv1_weight = cur_state_dict['conv1.weight']
        # cur_state_dict['conv1.weight'] = conv1_weight.sum(dim=1,keepdim=True)
        # model.load_state_dict(cur_state_dict)
        # print(model)



    model.to(**setup)
    model.eval()

    # Choose example images from the validation set or from third-party sources
    if args.num_images == 1:
        if args.target_id == -1:  # demo image

            if read_grey or change_inchannel:
                ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).resize(img_size))/255, **setup)
                print(ground_truth)
                ground_truth = ground_truth.view(1,1,*ground_truth.size())
                ground_truth = ground_truth.sub(xray_mean).div(xray_std)
                plt.imshow(tp(torch.cat((ground_truth,ground_truth,ground_truth),1)[0].cpu()))
                plt.show()

                img_shape = (1, ground_truth.shape[2], ground_truth.shape[3])
            else:
                # Specify PIL filter for lower pillow versions
                ground_truth = torch.as_tensor(np.array(Image.open(demo_img_path).convert('RGB').resize(img_size, Image.BICUBIC)) / 255, **setup)
                # print(ground_truth)
                ground_truth = ground_truth.permute(2, 0, 1).sub(dm).div(ds).unsqueeze(0).contiguous()

                img_shape = (3, ground_truth.shape[2], ground_truth.shape[3])

            print(ground_truth)
            print(ground_truth.shape)

            if not args.label_flip:
                labels = torch.as_tensor((img_label,), device=setup['device'])
            else:
                labels = torch.as_tensor((5,), device=setup['device'])
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

    else: # adapt this for multiple images without a predefined dataset
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
    if args.accumulation == 0: # one epoch

        model.zero_grad()
        if read_grey:
            target_loss, _, _ = loss_fn(model(torch.cat((ground_truth,ground_truth,ground_truth),1)), labels)
        else:
            target_loss, _, _ = loss_fn(model(ground_truth), labels)

        input_gradient = torch.autograd.grad(target_loss, model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]

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

            # why change this here? use xray mean std?
            # dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
            # ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
            ground_truth = ground_truth.to(**setup)
            input_gradient = [g.to(**setup) for g in input_gradient]
            model.to(**setup)
            model.eval()

        if args.optim == 'ours':
            # config = dict(signed=args.signed,
            #               boxed=args.boxed,
            #               cost_fn=args.cost_fn,
            #               indices='def',
            #               weights='equal',
            #               lr=0.1,
            #               optim=args.optimizer,
            #               restarts=args.restarts,
            #               max_iterations=1000,
            #               total_variation=args.tv,
            #               init='randn',
            #               filter='none',
            #               lr_decay=True,
            #               scoring_choice='loss')
            config = set_config

    #     elif args.optim == 'zhu':
    #         config = dict(signed=False,
    #                       boxed=False,
    #                       cost_fn='l2',
    #                       indices='def',
    #                       weights='equal',
    #                       lr=1e-4,
    #                       optim='LBFGS',
    #                       restarts=args.restarts,
    #                       max_iterations=300,
    #                       total_variation=args.tv,
    #                       init=args.init,
    #                       filter='none',
    #                       lr_decay=False,
    #                       scoring_choice=args.scoring_choice)
    #
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images)
        output, stats = rec_machine.reconstruct(input_gradient, labels, read_grey=read_grey, greyscale=greyscale, img_shape=img_shape, dryrun=args.dryrun)


    else:
        pass
        # investigate how this works

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

    # if greyscale:
        # output = output.sub(dm).div(ds)
    # Compute stats
    if greyscale:
        ground_truth = ground_truth_denormalized
        factor = 1
    else:
        factor=1/ds

    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    # feat_mse = (model(output) - model(ground_truth)).pow(2).mean().item()
    feat_mse = np.nan
    # test_psnr = inversefed.metrics.psnr(output, ground_truth, factor=1 / ds)
    test_psnr = inversefed.metrics.psnr(output.detach(), ground_truth, factor=factor)


    # Save the best image
    if args.save_image and not args.dryrun:
        os.makedirs(args.image_path, exist_ok=True)
        output_denormalized = torch.clamp(output * ds + dm, 0, 1)
        # rec_filename = (f'{validloader.dataset.classes[labels][0]}_{"trained" if args.trained_model else ""}'
                        # f'{args.model}_{args.cost_fn}-{args.target_id}.png')
        rec_filename = f'{args.name}_rec_img_idx{stats["best_exp"]}.png'
        torchvision.utils.save_image(output_denormalized, os.path.join(args.image_path, rec_filename))

        if not greyscale:
            gt_denormalized = torch.clamp(ground_truth * ds + dm, 0, 1)
        else:
            gt_denormalized = ground_truth
        # gt_filename = (f'{validloader.dataset.classes[labels][0]}_ground_truth-{args.target_id}.png')
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


    # save parameters
    # print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |")
    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} | PSNR: {test_psnr:4.2f} |")

    inversefed.utils.save_to_table(args.table_path, name=f'exp_{args.name}', dryrun=args.dryrun,

                                   model=args.model,
                                   dataset=args.dataset,
                                   trained=args.trained_model,
                                   accumulation=args.accumulation,
                                   restarts=args.restarts,
                                   OPTIM=args.optim,
                                   cost_fn=args.cost_fn,
                                   indices=args.indices,
                                   weights=args.weights,
                                   scoring=args.scoring_choice,
                                   init=args.init,
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
