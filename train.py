import time
from loss_function import *

def get_base_schedular(optimizer,initial_lr):
    initial_lr = initial_lr
    warm_up_step = 5
    mile_stone = [160,220]
    gamma = 0.01

    def get_lr(epoch):
        if epoch <= warm_up_step:
            return initial_lr * epoch / warm_up_step

        decrease_rate = 1
        for th in mile_stone:
            if epoch >= th:
                decrease_rate *= gamma
        return initial_lr*decrease_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer,get_lr)

def get_minimax_schedular(optimizer, initial_lr):
    initial_lr = initial_lr
    warm_up_step = 5
    mile_stone = [200,320]
    gamma = 0.01

    def get_lr(epoch):
        if epoch <= warm_up_step:
            return initial_lr * epoch / warm_up_step

        decrease_rate = 1
        for th in mile_stone:
            if epoch >= th:
                decrease_rate *= gamma
        return initial_lr*decrease_rate

    return torch.optim.lr_scheduler.LambdaLR(optimizer,get_lr)


def get_criterion(args,class_sample_num_dict,device):
    '''
    Most hyperparameters are from VS-loss paper for fair comparison.
    [1] G. R. Kini, O. Paraskevas, S. Oymak, and C. Thrampoulidis, “Labelimbalanced
    and group-sensitive classification under overparameterization,”
    Advances in Neural Information Processing Systems, vol. 34, pp. 18 970–
    18 983, 2021.
    '''
    loss_function_name = args.loss_function
    num_train_samples = class_sample_num_dict["num_train_samples"]

    if loss_function_name == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif loss_function_name == 'WCE':
        criterion = nn.CrossEntropyLoss(balanced_class_rate(num_train_samples).to(device))
    elif loss_function_name == 'Focal':
        criterion = FocalLoss()
    elif loss_function_name == 'Focal_alpha':
        criterion = FocalLoss(balanced_class_rate(num_train_samples).to(device))
    elif loss_function_name == 'LDAM':
        criterion = LDAMLoss(num_train_samples)
    elif loss_function_name == 'VS':
        if args.data_name == 'CIFAR10':
            if args.imbalance_type == 'step':
                gamma = 0.2
                tau = 1.5
            elif args.imbalance_type == 'LT':
                gamma = 0.15
                tau = 1.25
            else:
                raise ValueError("imbalance type does not match")

        elif args.data_name == 'CIFAR100':
            if args.imbalance_type == 'step':
                gamma = 0.05
                tau = 0.5
            elif args.imbalance_type == 'LT':
                gamma = 0.05
                tau = 0.75
            else:
                raise ValueError("imbalance type does not match")
        else:
            gamma = 0.3
            tau = 1.0

        if loss_function_name == 'VS':
            criterion = VSLoss(num_train_samples, gamma=gamma, tau=tau)
        else:
            raise NotImplementedError(f'{loss_function_name} did not implemented')


    elif loss_function_name == 'TWCE_EGA':
        criterion = TWCE_EGA(torch.tensor(num_train_samples).to(device),
                             device, alpha=args.prior_lr)
    elif loss_function_name == 'TWCE_linear_ascent':
      if args.data_name == 'CIFAR10' and args.imbalance_type == 'LT':
        group = 3
      else:
        group = 10

      criterion = TWCE_linear_ascent(torch.tensor(num_train_samples).to(device),
                                     device, alpha=args.prior_lr, group=group)

    elif loss_function_name == 'LA' or loss_function_name == 'TLA_EGA' \
         or loss_function_name == 'TLA_linear_ascent':
        # the value of hyperparameters are from original paper of VS loss(https://arxiv.org/abs/2103.01550).
        if args.data_name == 'CIFAR10':
            if args.imbalance_type == 'step':
                tau = 2.25
            elif args.imbalance_type == 'LT':
                tau = 2.25
            else:
                raise ValueError("imbalance type does not match")

        elif args.data_name == 'CIFAR100':
            if args.imbalance_type == 'step':
                tau = 0.875
            elif args.imbalance_type == 'LT':
                tau = 1.375
            else:
                raise ValueError("imbalance type does not match")
        else:
            # default value of LA Loss paper
            tau = 1.0

        if loss_function_name == 'LA':
            criterion = LALoss(num_train_samples,tau=tau)
        elif loss_function_name == 'TLA_EGA':
            criterion = TLA_EGA(num_train_samples, tau=tau, alpha=args.prior_lr)
        elif loss_function_name == 'TLA_linear_ascent':
            if args.data_name == 'CIFAR10' and args.imbalance_type == 'LT':
                group = 3
            elif args.data_name == 'CIFAR10' and args.imbalance_type == 'step' and args.imbalance_ratio == 0.1:
                group = 5
            else:
                group = 10
            criterion = TLA_linear_ascent(num_train_samples, tau=tau, alpha=args.prior_lr, group=group)
        else:
            raise NotImplementedError(f'{loss_function_name} did not implemented')

    else:
        raise ValueError("loss_function_name does not match")
    return criterion

def train_model(model, train_dataloader, test_dataloader,criterion, optimizer, num_epoch,
                device,num_classes,scheduler=None):
    model.to(device)
    since = time.time()
    result = []
    train_class_acc_info_list = []
    test_class_acc_info_list = []
    train_loss_info_list = []
    train_acc_info_list = []
    test_acc_info_list = []

    #num_epoch
    for epoch in range(num_epoch):
        train_class_acc_info = make_init_class_acc_info(num_classes)

        model.train()
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        train_epoch_loss = 0.0
        train_epoch_corrects = 0
        count = 0

        for x, labels in train_dataloader:

            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * inputs.size(0)
            train_epoch_corrects += torch.sum(preds == labels.data)
            train_class_acc_info = update_class_acc_info(train_class_acc_info,preds,labels)


        print("plain_train_count_per_epoch", count)
        train_epoch_loss = train_epoch_loss / count
        epoch_acc = train_epoch_corrects.double() / count
        train_class_acc_info_list.append(train_class_acc_info)

        train_loss_info_list.append(train_epoch_loss)
        train_acc_info_list.append(epoch_acc.item())

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, epoch_acc))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_loss)
            else:
                scheduler.step()

        model.eval()

        test_epoch_corrects = 0
        count = 0
        test_class_acc_info = make_init_class_acc_info(num_classes)

        for x, labels in test_dataloader:
            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_epoch_corrects += torch.sum(preds == labels.data)
            test_class_acc_info = update_class_acc_info(test_class_acc_info, preds, labels)

        epoch_acc = test_epoch_corrects.double() / count
        test_class_acc_info_list.append(test_class_acc_info)
        test_acc_info_list.append(epoch_acc.item())

        print('Test Acc: {:.4f}'.format(epoch_acc))
        print('-' * 20)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    result.append(refine_class_acc_info_list(train_class_acc_info_list))
    result.append(refine_class_acc_info_list(test_class_acc_info_list))
    result.append(train_loss_info_list)
    result.append(train_acc_info_list)
    result.append(test_acc_info_list)

    return result

# appropriately modified from https://github.com/kaidic/LDAM-DRW/tree/master to fit our setting
def train_model_DRW(model, train_dataloader, test_dataloader,criterion, optimizer, num_epoch,
                device,num_classes,class_sample_num_dict,scheduler=None):
    model.to(device)
    since = time.time()
    result = []
    train_class_acc_info_list = []
    test_class_acc_info_list = []
    train_loss_info_list = []
    train_acc_info_list = []
    test_acc_info_list = []

    num_train_samples = class_sample_num_dict["num_train_samples"]
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, num_train_samples)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(num_train_samples)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device)

    #num_epoch
    for epoch in range(num_epoch):
        if epoch > 160 :
            criterion.set_weight(per_cls_weights)

        train_class_acc_info = make_init_class_acc_info(num_classes)

        model.train()
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        train_epoch_loss = 0.0
        train_epoch_corrects = 0
        count = 0

        for x, labels in train_dataloader:

            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * inputs.size(0)
            train_epoch_corrects += torch.sum(preds == labels.data)
            train_class_acc_info = update_class_acc_info(train_class_acc_info,preds,labels)


        print("DRW_train_count_per_epoch", count)
        train_epoch_loss = train_epoch_loss / count
        epoch_acc = train_epoch_corrects.double() / count
        train_class_acc_info_list.append(train_class_acc_info)

        train_loss_info_list.append(train_epoch_loss)
        train_acc_info_list.append(epoch_acc.item())

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, epoch_acc))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_loss)
            else:
                scheduler.step()

        model.eval()

        test_epoch_corrects = 0
        count = 0
        test_class_acc_info = make_init_class_acc_info(num_classes)

        for x, labels in test_dataloader:
            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_epoch_corrects += torch.sum(preds == labels.data)
            test_class_acc_info = update_class_acc_info(test_class_acc_info, preds, labels)

        epoch_acc = test_epoch_corrects.double() / count
        test_class_acc_info_list.append(test_class_acc_info)
        test_acc_info_list.append(epoch_acc.item())

        print('Test Acc: {:.4f}'.format(epoch_acc))
        print('-' * 20)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    result.append(refine_class_acc_info_list(train_class_acc_info_list))
    result.append(refine_class_acc_info_list(test_class_acc_info_list))
    result.append(train_loss_info_list)
    result.append(train_acc_info_list)
    result.append(test_acc_info_list)

    return result

def train_minimax(model, data_load_dict, criterion, optimizer, num_epoch,
                   device, num_classes, scheduler):
    model.to(device)
    since = time.time()
    result = []
    train_class_acc_info_list = []
    test_class_acc_info_list = []
    train_loss_info_list = []
    train_acc_info_list = []
    test_acc_info_list = []
    validation_class_acc_info_list = []
    pi_t_info_list = []
    test_class_acc_info_at_val_list = []


    train_loader = data_load_dict["train_loader"]
    validation_loader = data_load_dict["val_loader"]
    test_dataloader = data_load_dict["test_loader"]

    total_train_dataloader = data_load_dict["total_loader"]


    #---------------------------------------------------------#
    #update phase
    print("update phase")
    #num_epoch
    for epoch in range(num_epoch):

        train_class_acc_info = make_init_class_acc_info(num_classes)

        model.train()
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))
        print('-' * 20)

        train_epoch_loss = 0.0
        train_epoch_corrects = 0
        count = 0

        for x, labels in train_loader:

            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * inputs.size(0)
            train_epoch_corrects += torch.sum(preds == labels.data)
            train_class_acc_info = update_class_acc_info(train_class_acc_info,preds,labels)

        print("minimax_train_count_per_epoch", count)
        train_epoch_loss = train_epoch_loss / count
        epoch_acc = train_epoch_corrects.double() / count
        train_class_acc_info_list.append(train_class_acc_info)

        train_loss_info_list.append(train_epoch_loss)
        train_acc_info_list.append(epoch_acc.item())

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, epoch_acc))

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_loss)
            else:
                scheduler.step()

        model.eval()
        
        if epoch > 5:

            validation_class_acc_info = make_init_class_acc_info(num_classes)
            for x, labels in validation_loader:
                inputs = x
                count += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                validation_class_acc_info = update_class_acc_info(validation_class_acc_info, preds, labels)

            validation_class_acc_info_list.append(validation_class_acc_info)
            pi_t_info_list.append(criterion.pi_t.tolist())
            criterion.update_parameter(validation_class_acc_info)
            
            
        test_epoch_corrects = 0
        count = 0
        test_class_acc_info = make_init_class_acc_info(num_classes)

        for x, labels in test_dataloader:
            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_epoch_corrects += torch.sum(preds == labels.data)
            test_class_acc_info = update_class_acc_info(test_class_acc_info, preds, labels)

        epoch_acc = test_epoch_corrects.double() / count
        test_class_acc_info_list.append(test_class_acc_info)
        test_acc_info_list.append(epoch_acc.item())
        
        if  epoch > 5:
          test_class_acc_info_at_val_list.append(test_class_acc_info)

        print('Test Acc: {:.4f}'.format(epoch_acc))
        print('-' * 20)


    # ---------------------------------------------------------#
    pi_t_info_list.append(criterion.pi_t.tolist())

    # learning with fixed criterion parameter
    print("learning with fixed criterion parameter")
    #30
    for epoch in range(30):
        train_class_acc_info = make_init_class_acc_info(num_classes)

        model.train()
        print('Epoch {}/{}'.format(epoch + 1 + num_epoch, 30+num_epoch))
        print('-' * 20)

        train_epoch_loss = 0.0
        train_epoch_corrects = 0
        count = 0

        for x, labels in total_train_dataloader:
            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item() * inputs.size(0)
            train_epoch_corrects += torch.sum(preds == labels.data)
            train_class_acc_info = update_class_acc_info(train_class_acc_info, preds, labels)

        print("minimax_train_count_per_epoch", count)
        train_epoch_loss = train_epoch_loss / count
        epoch_acc = train_epoch_corrects.double() / count
        train_class_acc_info_list.append(train_class_acc_info)

        train_loss_info_list.append(train_epoch_loss)
        train_acc_info_list.append(epoch_acc.item())

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_epoch_loss)
            else:
                scheduler.step()

        print('Train Loss: {:.4f} Acc: {:.4f}'.format(train_epoch_loss, epoch_acc))

        model.eval()

        test_epoch_corrects = 0
        count = 0
        test_class_acc_info = make_init_class_acc_info(num_classes)

        for x, labels in test_dataloader:
            inputs = x
            count += inputs.shape[0]

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_epoch_corrects += torch.sum(preds == labels.data)
            test_class_acc_info = update_class_acc_info(test_class_acc_info, preds, labels)

        epoch_acc = test_epoch_corrects.double() / count
        test_class_acc_info_list.append(test_class_acc_info)
        test_acc_info_list.append(epoch_acc.item())

        print('Test Acc: {:.4f}'.format(epoch_acc))
        print('-' * 20)

    #---------------------------------------------------------#
    #Training completed
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    result.append(refine_class_acc_info_list(train_class_acc_info_list))
    result.append(refine_class_acc_info_list(test_class_acc_info_list))
    result.append(train_loss_info_list)
    result.append(train_acc_info_list)
    result.append(test_acc_info_list)
    result.append(refine_class_acc_info_list(validation_class_acc_info_list))
    result.append(pi_t_info_list)
    result.append(refine_class_acc_info_list(test_class_acc_info_at_val_list))

    return result




