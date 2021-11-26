import torch
import torch.nn as nn


class GANLossObj(nn.Module):
    """
    Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.
    Because we need to calculate the loss between the output vector of the discriminator and True/False, this class
    automatically extend True/False to an all_1 vector or an all_0 vector so the loss function is able to calculate.
    """
    def __init__(self, GAN_mode, real_label=1.0, fake_label=0.0):
        """
        Initialize the GANLossObj class.

        Parameters:
            GAN_mode (str)        -- The type of GAN objective. Vanilla GAN loss is the cross-entropy objective used in the original GAN paper.
            real_label (float)    -- the label value for the real omics data
            fake_label (float)    -- the label value for the synthetic omics data

        Note:
            Do not use sigmoid as the last layer of Discriminator.
            LSGAN needs no sigmoid. Vanilla GANs will handle it with BCEWithLogitsLoss.
            This is why we don't put a sigmoid layer in the discriminator
        """
        super(GANLossObj, self).__init__()
        # registering the labels for model storage
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.GAN_mode = GAN_mode
        # Determine the loss function
        if GAN_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif GAN_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif GAN_mode == 'wgangp':
            self.loss = None
        else:
            raise NotImplementedError('GAN mode %s is not found' % GAN_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensors with the same size as the input tensor

        Parameters:
            prediction (tensor)     -- the output from a discriminator
            target_is_real (bool)   -- whether the ground truth label is for real or fake

        Returns:
            A label tensor filled with ground truth label with the size of the input tensor
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        Calculate the loss given output from the discriminator and the grount truth tensor.

        Parameters:
            prediction (tensor)     -- the output from a discriminator
            target_is_real (bool)   -- whether the ground truth label is for real or fake

        Returns:
            The calculated loss
        """
        if self.GAN_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.GAN_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise NotImplementedError('GAN mode %s is not found' % self.GAN_mode)
        return loss


def get_dist_loss(loss_name, reduction='mean'):
    """
    Return the distance loss function.
    Parameters:
        loss_name (str)    -- the name of the loss function: BCE | MSE | L1 | CE
        reduction (str)    -- the reduction method applied to the loss function: sum | mean
    """
    if loss_name == 'BCE':
        return nn.BCELoss(reduction=reduction)
    elif loss_name == 'MSE':
        return nn.MSELoss(reduction=reduction)
    elif loss_name == 'L1':
        return nn.L1Loss(reduction=reduction)
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss(reduction=reduction)
    else:
        raise NotImplementedError('Loss function %s is not found' % loss_name)


def kl_loss(mean, log_var, reduction='mean'):
    part_loss = 1 + log_var - mean.pow(2) - log_var.exp()
    if reduction == 'mean':
        loss = -0.5 * torch.mean(part_loss)
    else:
        loss = -0.5 * torch.sum(part_loss)
    return loss