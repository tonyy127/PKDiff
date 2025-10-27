"""Evaluates the performance of a model"""
import logging
import math
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex, F1Score
from tqdm import tqdm

from utils1 import *
# from utils.cityscapes_loader import decode_segmap as decode_segmap_cityscapes
from utils.utils import diffuse, denoise_scale
# from utils.uavid_loader import decode_segmap as decode_segmap_uavid, UAVidLoader
# from utils.vaihingen_buildings_loader import decode_segmap as decode_segmap_vaihingen



def segmentation_dice(predicted_probs, target_segmentation, smooth=1.0):
    """Returns Dice Loss for multi-class segmentation.
    predicted_probs: (B, C, H, W) 的 softmax 后概率
    target_segmentation: (B, H, W) 的 long 类型类别索引
    """
    # One-hot encode the target
    num_classes = predicted_probs.shape[1]
    target_one_hot = F.one_hot(target_segmentation, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Compute Dice for each class and average
    intersection = torch.sum(predicted_probs * target_one_hot, dim=(2, 3))
    union = torch.sum(predicted_probs, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
    dice_per_class = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice_per_class.mean(dim=1).mean()  # Average over classes and batch
    
    return dice_loss

def segmentation_cross_entropy(predicted_segmentation, target_segmentation):
    """Returns Cross Entropy Loss"""
    weights = torch.tensor([1.79, 1.0, 2.17, 1.17, 3.2, 27.14, 21.56, 190.25], dtype=torch.float32).to(target_segmentation.device)
    criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss = criterion(predicted_segmentation, target_segmentation)
    return loss

def noise_mse(noise_predicted, noise_target):
    """Returns MSE Loss"""
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(noise_predicted, noise_target)
    return loss

def compute_total_loss(segmentation_cross_entropy):
    """Returns total loss"""
    total_loss =  (1 * segmentation_cross_entropy)
    return total_loss

def write_images_to_tensorboard(writer, epoch, image=None, seg_diffused=None, seg_predicted=None, seg_gt=None, datasplit='validation', dataset_name='cityscapes'):
        """Writes images to TensorBoard"""
        # decode segmap based on dataset
        if dataset_name == 'cityscapes':
            decode_segmap = decode_segmap_cityscapes
        elif dataset_name == 'uavid':
            decode_segmap = decode_segmap_uavid
        elif dataset_name == 'vaihingen':
            decode_segmap = decode_segmap_vaihingen
        else:
            raise NotImplementedError('Dataset {} not implemented'.format(dataset_name))
        if image is not None:
            image = torchvision.utils.make_grid(image, normalize=True) # normalize to [0,1] and convert to uint8
            writer.add_images('{}/image'.format(datasplit), image, epoch, dataformats='CHW')
        if seg_diffused is not None:
            seg_diffused = decode_segmap(seg_diffused, is_one_hot=True)
            writer.add_images('{}/seg_diffused'.format(datasplit), seg_diffused, epoch, dataformats='CHW')
        if seg_predicted is not None:
            seg_predicted = decode_segmap(seg_predicted, is_one_hot=True)
            writer.add_images('{}/seg_predicted'.format(datasplit), seg_predicted, epoch, dataformats='CHW')
        if seg_gt is not None:
            seg_gt = decode_segmap(seg_gt, is_one_hot=False)
            writer.add_images('{}/seg_gt'.format(datasplit), seg_gt, epoch, dataformats='CHW')

def denoise_loop_scales(model, device, network_config, images):
    """Denoises all scales for a single timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Initialize built in ensemble
    seg_denoised_ensemble = torch.zeros(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        
        for scale in range(network_config.n_scales): # for each scale
            # Resize to current scale
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False).softmax(dim=1)

            # Diffuse
            t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
            seg_diffused = diffuse(seg_previous_scaled, t)  
            # Denoise
            seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised
        
        # Add to ensemble
        if network_config.built_in_ensemble:
            if timestep == 0:
                seg_denoised_ensemble = seg_denoised
            else:
                seg_denoised_ensemble = seg_denoised_ensemble / 2 + seg_denoised / 2
            
            seg_previous_scaled = seg_denoised_ensemble
    
    return seg_denoised

def denoise_linear_scales(model, device, network_config, images):
    """Denoises one scale at a each timestep"""
    # Calculate scale sizes (smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # Initialize first prediction (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        # Get the current scale
        timesteps_per_scale = math.ceil(network_config.n_timesteps / network_config.n_scales)
        scale = timestep // timesteps_per_scale
        
        # Resize to current scale
        if timestep % timesteps_per_scale == 0:
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

        # Diffuse
        t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
        seg_diffused = diffuse(seg_previous_scaled, t)
        # Denoise
        seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

        # Update the previous segmentation map
        seg_previous_scaled = seg_denoised

    return seg_denoised

def denoise(model, device, network_config, images):
        """Denoises the segmentation map"""
        if network_config.scale_procedure == 'loop':
            seg_denoised = denoise_loop_scales(model, device, network_config, images)
        elif network_config.scale_procedure == 'linear':
            seg_denoised = denoise_linear_scales(model, device, network_config, images)

        return seg_denoised

class Evaluator:
    """Evaluates the performance of a model"""
    def __init__(self, model, network_config, device, dataset_selection=None, test_data_loader=None, validation_data_loader=None, writer=None):
        self.model = model
        self.network_config = network_config
        self.device = device
        self.dataset_selection = dataset_selection
        self.test_data_loader = test_data_loader
        self.validation_data_loader = validation_data_loader
        self.writer = writer

    def evaluate(self, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE,
                 epoch=1, is_test=True, ensemble=1): # epoch=None
        """Evaluates the model on the given dataset"""
        model = self.model
        network_config = self.network_config
        # model.eval()

        # Use the network on the test set
        ## Potsdam
        if DATASET == 'Potsdam':
            test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, :3], dtype='float32') for id in
                           test_ids)
            # test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id))[:, :, (3, 0, 1, 2)][:, :, :3], dtype='float32') for id in test_ids)
        ## Vaihingen
        else:
            test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_dsms = (np.asarray(io.imread(DSM_FOLDER.format(id)), dtype='float32') for id in test_ids)
        test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
        eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
        all_preds = []
        all_gts = []

        # Switch the network to inference mode
        with torch.no_grad():
            for img, dsm, gt, gt_e in tqdm(zip(test_images, test_dsms, test_labels, eroded_labels), total=len(test_ids),
                                           leave=False):
                pred = np.zeros(img.shape[:2] + (N_CLASSES,))

                total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
                for i, coords in enumerate(
                        tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                             total=total,
                             leave=False)):
                    # Build the tensor
                    image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
                    image_patches = np.asarray(image_patches)
                    image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

                    min = np.min(dsm)
                    max = np.max(dsm)
                    dsm = (dsm - min) / (max - min)
                    dsm_patches = [np.copy(dsm[x:x + w, y:y + h]) for x, y, w, h in coords]
                    dsm_patches = np.asarray(dsm_patches)
                    dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)

                    outs = denoise(model, self.device, network_config, image_patches)
                    outs = outs.data.cpu().numpy()

                    # Fill in the results array
                    for out, (x, y, w, h) in zip(outs, coords):
                        out = out.transpose((1, 2, 0))
                        pred[x:x + w, y:y + h] += out
                    del (outs)

                pred = np.argmax(pred, axis=-1)
                all_preds.append(pred)
                all_gts.append(gt_e)

        accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
        if all:
            return accuracy, all_preds, all_gts
        else:
            return accuracy
        # # Ensamble
        # for i in range(ensemble-1):
        #     seg_denoised += denoise(model, self.device, network_config, image_patches)
        # seg_denoised /= ensemble

        # # Compute loss
        # seg_predicted = seg_denoised.view(seg_denoised.shape[0], seg_denoised.shape[1], -1).argmax(dim=1)
        # seg_target = seg_gt.view(seg_gt.shape[0], -1)
        # jaccard_index.update(seg_predicted, seg_target)
        # jaccard_per_class.update(seg_predicted, seg_target)
        # f1_score.update(seg_predicted, seg_target)

        # # Write images to tensorboard
        # if self.writer is not None:
        #     if it < 8:
        #         write_images_to_tensorboard(self.writer, epoch, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0], datasplit='validation/{}'.format(it))


        # # Overall metrics
        # jaccard_index_total = jaccard_index.compute()
        # jaccard_per_class_total = jaccard_per_class.compute()
        # f1_score_total = f1_score.compute()
        #
        # # Text report
        # report = 'Jaccard index: {:.4f} | F1 score: {:.4f}'.format(jaccard_index_total, f1_score_total)
        # report_per_class = 'Jaccard index per class: {}'.format(jaccard_per_class_total)
        # # if self.writer is None:
        # #     logging.log(logging.WARNING, report)
        # #     logging.log(logging.WARNING, report_per_class)
        # else:
        #     logging.info('{} | {} | {}'.format("Test" if is_test else "Validation | Epoch: {}".format(epoch), report, report_per_class))
        #
        # # Write to tensorboard
        # if self.writer is not None:
        #     self.writer.add_scalar('{}/JaccardIndex'.format('test' if is_test else 'validation'), jaccard_index_total, epoch)
        #     self.writer.add_scalar('{}/F1Score'.format('test' if is_test else 'validation'), f1_score_total, epoch)
        
    def validate(self, epoch):
        """Evaluates the model on the validation dataset"""
        self.evaluate(self.validation_data_loader, epoch, is_test=False)

    def test(self, ensemble=1):
        """Evaluates the model on the test dataset"""
        self.evaluate(self.test_data_loader, is_test=True, ensemble=ensemble)

