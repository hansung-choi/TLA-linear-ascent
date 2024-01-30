from typing import Optional
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from utils import *

'''
Reference:
    Focal loss
    [1] T. Y. Lin, P. Goyal, R. Girshick, K. He, and P. Doll´ar, “Focal loss for
    dense object detection,” IEEE International Conference on Computer
    Vision, pp. 2980–2988, 2017.

    LDAM loss
    [2] K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, “Learning
    imbalanced datasets with label-distribution-aware margin loss,” Advances
    in Neural Information Processing Systems, vol. 32, pp. 1567–1578, 2019.
    
    LA loss
    [3] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar,
    “Long-tail learning via logit adjustment,” International Conference on
    Learning Representations, 2021.
    
    VS loss
    [4] G. R. Kini, O. Paraskevas, S. Oymak, and C. Thrampoulidis, “Labelimbalanced
    and group-sensitive classification under overparameterization,”
    Advances in Neural Information Processing Systems, vol. 34, pp. 18 970–
    18 983, 2021.
    
    TWCE and EGA method
    [5] R. Alaiz-Rodr´ıguez, A. Guerrero-Curieses, and J. Cid-Sueiro, “Minimax
    classifiers based on neural networks,” Pattern Recognition, vol. 38, no. 1,
    pp. 29–39, 2005.
    [6] R. Alaiz-Rodrıguez, A. Guerrero-Curieses, and J. Cid-Sueiro, “Minimax
    regret classifier for imprecise class distributions,” Journal of Machine
    Learning Research, vol. 8, pp. 103–130, 2007.
    [7] S. Sagawa, P. W. Koh, T. B. Hashimoto, and P. Liang, “Distributionally
    robust neural networks for group shifts: On the importance of regularization
    for worst-case generalization,” International Conference on Learning
    Representations, 2020.
    [8] J. Zhang, A. Menon, A. Veit, S. Bhojanapalli, S. Kumar, and S. Sra, “Coping
    with label shift via distributionally robust optimisation,” International
    Conference on Learning Representations, 2021.
'''


class FocalLoss(nn.Module):
    def __init__(self,
                 class_weight: Optional[Tensor] = None,
                 gamma: float=2,
                 reduction: str = 'mean'):
        super().__init__()
        self.class_weight = class_weight
        self.gamma = gamma

        if reduction not in ['mean','sum','none']:
            raise ValueError(
                'reduction should be mean or sum or none.'
            )

        self.reduction = reduction
        self.nill_loss = nn.NLLLoss(weight=class_weight,reduction='none')

    def forward(self,x : Tensor, y : Tensor) -> Tensor:
        #x should be NxC shape tensor.

        #calculate CE part
        y = y.view(-1)
        log_p = F.log_softmax(x,dim= -1)
        ce_part = self.nill_loss(log_p,y)

        #calculate (1-pt)^(gamma) part
        c = x.shape[1]
        one_hot_label_matrix = F.one_hot(y,num_classes=c)
        valid_log_p = log_p*one_hot_label_matrix
        log_pt = valid_log_p.sum(1)
        pt = log_pt.exp()
        focal_part = (1-pt)**self.gamma

        loss = focal_part*ce_part

        if self.reduction == 'mean':
            devide_term = x.shape[0]
            if self.class_weight != None:
                devide_term = (self.class_weight*one_hot_label_matrix).sum()
            loss = loss.sum()/devide_term

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

    def set_weight(self,weight):
        self.weight = weight

class LALoss(nn.Module):

    def __init__(self, cls_num_list, tau=1.0, weight=None):
        super(LALoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]


        iota_list = tau * np.log(cls_probs)


        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.weight = weight

    def forward(self, x, target):
        output = x + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = weight

    def forward(self, x, target):
        output = x / self.Delta_list + self.iota_list

        return F.cross_entropy(output, target, weight=self.weight)

class TWCE_EGA(nn.Module):
    def __init__(self,
                 prior_distribution: Tensor,
                 device,
                 alpha: float = 0.1,
                 reduction: str = 'mean'):
        super().__init__()
        #calculate desired class weight based on full batch prior distribution
        self.prior_distribution = prior_distribution/prior_distribution.sum()
        self.device = device
        self.alpha = alpha
        self.num_classes = prior_distribution.shape[0]

        if reduction not in ['mean','sum','none']:
            raise ValueError(
                'reduction should be mean or sum or none.'
            )

        self.reduction = reduction

        #info for updating desired class weight
        self.train_class_acc_info = make_init_class_acc_info(self.num_classes)

        self.desired_data_prior = self.prior_distribution.clone().detach()
        self.pi_t = self.prior_distribution.clone().detach().cpu().numpy()

        desired_class_weight = self.desired_data_prior / self.prior_distribution
        self.criterion = nn.CrossEntropyLoss(desired_class_weight, reduction=self.reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        #x should be NxC shape tensor.

        loss = self.criterion(x, y)

        return loss


    def update_parameter(self,train_class_acc_info):
        #update_disired_class_weight
        # pi_t+1 = pi_t * exp(alpha*error) #alpha: 0.1
        self.train_class_acc_info = train_class_acc_info
        acc_per_class = refine_class_acc_info(train_class_acc_info)
        acc_per_class = np.array(acc_per_class)
        error_per_class = 100.0 - acc_per_class
        error_probability_per_class = error_per_class/100

        pi_t = self.desired_data_prior.clone().detach().cpu().numpy()

        pi_new = pi_t * np.exp(self.alpha*(error_probability_per_class))
        pi_new = pi_new / np.sum(pi_new)


        self.desired_data_prior = torch.FloatTensor(pi_new).cuda()
        self.desired_data_prior = self.desired_data_prior / self.desired_data_prior.sum()
        self.pi_t = self.desired_data_prior.clone().detach().cpu().numpy()

        desired_class_weight = self.desired_data_prior / self.prior_distribution

        self.criterion = nn.CrossEntropyLoss(desired_class_weight)

class TWCE_linear_ascent(nn.Module):
    def __init__(self,
                 prior_distribution: Tensor,
                 device,
                 alpha: float = 0.01,
                 group: int = 10,
                 reduction: str = 'mean'):
        super().__init__()
        # calculate desired class weight based on full batch prior distribution
        self.prior_distribution = prior_distribution / prior_distribution.sum()
        self.device = device
        self.alpha = alpha
        self.num_classes = prior_distribution.shape[0]
        self.group = group
        self.k = int(self.num_classes//self.group)  # num classes per group

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(
                'reduction should be mean or sum or none.'
            )

        self.reduction = reduction

        # info for updating desired class weight
        self.train_class_acc_info = make_init_class_acc_info(self.num_classes)

        self.desired_data_prior = self.prior_distribution.clone().detach()
        self.pi_t = self.prior_distribution.clone().detach().cpu().numpy()
        desired_class_weight = self.desired_data_prior / self.prior_distribution

        self.criterion = nn.CrossEntropyLoss(desired_class_weight, reduction=self.reduction)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # x should be NxC shape tensor.

        loss = self.criterion(x, y)

        return loss

    def update_parameter(self, train_class_acc_info):
        # update_disired_class_weight
        self.train_class_acc_info = train_class_acc_info
        current_max_error_class_list = find_k_max_error_class(self.train_class_acc_info,self.k)

        max_error_data_prior = torch.FloatTensor([0. for i in range(self.num_classes)]).to(self.device)
        for current_k_max_error_class in current_max_error_class_list:
            max_error_data_prior[current_k_max_error_class] = 1./self.k

        feasible_direction = max_error_data_prior - self.desired_data_prior
        self.desired_data_prior = self.desired_data_prior + self.alpha * feasible_direction
        self.desired_data_prior = self.desired_data_prior / self.desired_data_prior.sum()
        self.pi_t = self.desired_data_prior.clone().detach().cpu().numpy()

        desired_class_weight = self.desired_data_prior / self.prior_distribution

        self.criterion = nn.CrossEntropyLoss(desired_class_weight)

class TLA_EGA(nn.Module):

    def __init__(self, cls_num_list, tau=1.0, alpha=0.1,weight=None):
        super(TLA_EGA, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        self.pi_t = np.array(cls_probs)

        self.tau = tau
        self.alpha = alpha
        iota_list = np.log(cls_probs)


        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.robust_list = self.iota_list.clone().detach()
        self.weight = weight

    def forward(self, x, target):
        output = x + self.tau * (self.iota_list - self.robust_list)

        return F.cross_entropy(output, target, weight=self.weight)

    def update_parameter(self,train_class_acc_info):
        #update_disired_class_weight
        # pi_t+1 = pi_t * exp(alpha*error) #alpha: 0.1
        acc_per_class = refine_class_acc_info(train_class_acc_info)
        acc_per_class = np.array(acc_per_class)
        error_per_class = 100.0 - acc_per_class
        error_probability_per_class = error_per_class/100

        pi_t = np.exp(self.robust_list.clone().detach().cpu().numpy())

        pi_new = pi_t * np.exp(self.alpha*(error_probability_per_class))
        pi_new = pi_new / np.sum(pi_new)
        self.pi_t = np.array(pi_new)

        self.robust_list = torch.FloatTensor(np.log(pi_new)).cuda()

class TLA_linear_ascent(nn.Module):

    def __init__(self, cls_num_list, tau=1.0, alpha=0.01,group: int = 10,weight=None):
        super(TLA_linear_ascent, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        self.pi_t = np.array(cls_probs)

        self.tau = tau
        self.alpha = alpha
        self.group = group
        self.num_classes = len(cls_probs)
        self.k = int(self.num_classes // self.group)
        iota_list = np.log(cls_probs)


        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.robust_list = self.iota_list.clone().detach()
        self.weight = weight

    def forward(self, x, target):
        output = x + self.tau * (self.iota_list - self.robust_list)

        return F.cross_entropy(output, target, weight=self.weight)

    def update_parameter(self,train_class_acc_info):
        #update_disired_class_weight
        # pi_t+1 = pi_t + alpha*(max_error_data_prior - pi_t) #alpha: 0.01
        current_max_error_class_list = find_k_max_error_class(train_class_acc_info, self.k)

        max_error_data_prior = np.array([0. for i in range(self.num_classes)])
        for current_k_max_error_class in current_max_error_class_list:
            max_error_data_prior[current_k_max_error_class] = 1. / self.k

        pi_t = np.exp(self.robust_list.clone().detach().cpu().numpy())

        feasible_direction = max_error_data_prior - pi_t
        pi_new = pi_t + self.alpha * feasible_direction
        pi_new = pi_new / np.sum(pi_new)
        self.pi_t = np.array(pi_new)

        self.robust_list = torch.FloatTensor(np.log(pi_new)).cuda()


























