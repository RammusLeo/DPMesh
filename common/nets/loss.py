import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from config import cfg
import torch.distributed as dist

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, student_feat, teacher_feat):
        loss = F.mse_loss(student_feat, teacher_feat, reduction='none')
        return loss

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D=None):
        assert coord_out.size() == coord_gt.size()
        loss = F.l1_loss(coord_out, coord_gt, reduction='none')
        loss = loss * valid
        # loss = torch.abs(coord_out - coord_gt) * valid
        if is_3D is not None:
            loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
            loss = torch.cat((loss[:,:,:2], loss_z),2)

        return loss

class VQLoss(nn.Module):
    def __init__(self) -> None:
        super(VQLoss, self).__init__()
    
    def forward(self, gt_indices, logits,classes, valid):
        bs = valid.shape[0]
        gt_indices = gt_indices. reshape(bs,-1)[valid==1]
        gt_indices = gt_indices.to(torch.int64)
        logits = logits.reshape(bs,-1,classes)[valid==1]
        loss = F.cross_entropy(logits.reshape(-1,classes).float(), gt_indices.reshape(-1), reduction='none')
        return loss

class SimCLR(nn.Module):
    def __init__(self, temperature=1.0):
        super(SimCLR, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch

        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        if len(z_i.shape) > 3:
            z_i = z_i.flatten(1,-1)
            z_j = z_j.flatten(1,-1)
        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity, batch_size).to(z_i)
        sim_ji = torch.diag(similarity, -batch_size).to(z_i)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(z_i)
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss


class ConfCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, confidence, is_3D):
        # confidence N, J
        # joint_valid N, J*3
        # batch_size = confidence.shape[0]
        # joint_valid = joint_valid.view(batch_size, -1, 3)[:, :, 2]  # N, J

        loss = (-torch.log(confidence + 1e-6) * is_3D[:,None,None].float()).mean()

        return loss

class ParamLoss(nn.Module):
    def __init__(self):
        super(ParamLoss, self).__init__()

    def forward(self, param_out, param_gt, valid):
        assert param_out.size() == param_gt.size()
        loss = F.l1_loss(param_out, param_gt, reduction='none')
        loss = loss * valid
        # loss = torch.abs(param_out - param_gt) * valid
        return loss

class NormalVectorLoss(nn.Module):
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:,face[:,1],:] - coord_out[:,face[:,0],:]
        v1_out = F.normalize(v1_out, p=2, dim=2) # L2 normalize to make unit vector
        v2_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,0],:]
        v2_out = F.normalize(v2_out, p=2, dim=2) # L2 normalize to make unit vector
        v3_out = coord_out[:,face[:,2],:] - coord_out[:,face[:,1],:]
        v3_out = F.normalize(v3_out, p=2, dim=2) # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:,face[:,1],:] - coord_gt[:,face[:,0],:]
        v1_gt = F.normalize(v1_gt, p=2, dim=2) # L2 normalize to make unit vector
        v2_gt = coord_gt[:,face[:,2],:] - coord_gt[:,face[:,0],:]
        v2_gt = F.normalize(v2_gt, p=2, dim=2) # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2) # L2 normalize to make unit vector

        valid_mask = valid[:,face[:,0],:] * valid[:,face[:,1],:] * valid[:,face[:,2],:]
        
        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) * valid_mask
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) * valid_mask
        loss = torch.cat((cos1, cos2, cos3),1)
        return loss

class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt, valid):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,1],:])**2,2,keepdim=True))
        d2_out = torch.sqrt(torch.sum((coord_out[:,face[:,0],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))
        d3_out = torch.sqrt(torch.sum((coord_out[:,face[:,1],:] - coord_out[:,face[:,2],:])**2,2,keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,1],:])**2,2,keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,0],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:,face[:,1],:] - coord_gt[:,face[:,2],:])**2,2,keepdim=True))

        valid_mask_1 = valid[:,face[:,0],:] * valid[:,face[:,1],:]
        valid_mask_2 = valid[:,face[:,0],:] * valid[:,face[:,2],:]
        valid_mask_3 = valid[:,face[:,1],:] * valid[:,face[:,2],:]
        
        diff1 = torch.abs(d1_out - d1_gt) * valid_mask_1
        diff2 = torch.abs(d2_out - d2_gt) * valid_mask_2
        diff3 = torch.abs(d3_out - d3_gt) * valid_mask_3
        loss = torch.cat((diff1, diff2, diff3),1)
        return loss

class DINOLoss(nn.Module):
    def __init__(self, out_dim, nepochs, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=0, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()

        # total_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
