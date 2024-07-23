import torchvision.transforms as transforms
import argparse
import torchvision
from model import *
from file_maker import *
from data_maker import *
from train import *

# data setting
# The name of data set
data_name_list = ["CIFAR10","CIFAR100"]

# step imbalance, LT(long tail imbalance)
imbalance_type_list = ["LT","step"]

# imbalance ratio
imbalance_ratio_list = [0.2,0.1,0.01]

# Train setting
# train_method
train_method_list = ['plain','DRW','minimax']

# model name
model_name_list = ['Resnet32']

# loss function

plain_loss_function_list = ['CE','WCE','Focal','Focal_alpha','LDAM','LA','VS','GML']
minimax_loss_function_list = ['TWCE_EGA','TWCE_linear_ascent',
                                    'TLA_EGA','TLA_linear_ascent']
loss_function_list = plain_loss_function_list + minimax_loss_function_list

# terminal run example:
# CUDA_VISIBLE_DEVICES=0 nohup python3 run_simulation.py -s 1 -d CIFAR10 -a simple_data_aug
# -t step -r 0.1 --train_method plain --loss_function CE &

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='simulation_result')

    # simulation number
    parser.add_argument('-s', '--simulation_number', default=1, type=int,
                        help='number to indicate which simulation it is')


    # data_setting
    parser.add_argument('-d', '--data_name', choices =data_name_list,required=True)
    parser.add_argument('-t', '--imbalance_type', choices=imbalance_type_list, required=True)
    parser.add_argument('-r', '--imbalance_ratio', type=float, choices=imbalance_ratio_list, required=True)

    # Train setting
    parser.add_argument('--train_method', choices=train_method_list, required=True)
    parser.add_argument('--model_name', choices =model_name_list, default='Resnet32')
    parser.add_argument('--loss_function', choices= loss_function_list, required=True)

    parser.add_argument('--num_workers', default=4, type=int, help = 'num_workers of data loader')
    parser.add_argument('--epochs', default=300, type=int, help='baseline training epoch')
    parser.add_argument('--batch_size', default=128, type=int, help='mini batch size')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate for model')
    parser.add_argument('--momentum', default=0.9, type=float,help='momentum')
    parser.add_argument('--weight_decay', default=2e-4, type=float, help='multiplier for L2 regularization')


    args = parser.parse_args()
    args.result_folder_name = '_'.join([args.model_name,args.data_name,args.loss_function,args.train_method,
                                        args.imbalance_type,str(args.imbalance_ratio),"sim_num",
                                        str(args.simulation_number)])
    result_folder_maker(args)

    # for reproducible simulation
    random_seed_num = int(args.simulation_number)
    torch.manual_seed(random_seed_num)
    np.random.seed(random_seed_num)
    random.seed(random_seed_num)


    # set data transform
    transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ])

    args.transform_train = transform_train
    args.transform_test = transform_test

    # download dataset
    if args.data_name == "CIFAR10":
        train_data_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=args.transform_train)
        test_data_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=args.transform_test)
        args.num_classes = 10
        args.per_class_size = 5000

    elif args.data_name == "CIFAR100":
        train_data_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=args.transform_train)
        test_data_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                               download=True, transform=args.transform_test)
        args.num_classes = 100
        args.per_class_size = 500

    else:
        raise ValueError("data_name does not match")

    args.train_data_set = train_data_set
    args.test_data_set = test_data_set

    # make data loader
    args.validation_ratio = 0.2
    data_load_dict, class_sample_num_dict = make_data_loaders(args)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make model
    use_norm = (args.loss_function =='LDAM')

    if args.model_name == 'Resnet32':
        model = Resnet32(color_channel = 3,num_classes=args.num_classes,use_norm= use_norm)
    else:
        raise ValueError("model_name does not match")
    model.to(device)

    # make optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # make scheduler
    if args.train_method == 'plain' or args.train_method == 'DRW':
        scheduler = get_base_schedular(optimizer,args.learning_rate)
    elif args.train_method == 'minimax':
        scheduler = get_minimax_schedular(optimizer, args.learning_rate)

    # make criterion
    if args.loss_function == 'TWCE_EGA' or args.loss_function == 'TLA_EGA':
        args.prior_lr = 0.1
    elif args.loss_function == 'TWCE_linear_ascent' or args.loss_function == 'TLA_linear_ascent':
        args.prior_lr = 0.01

    criterion = get_criterion(args,class_sample_num_dict,device)

    # model train and get its result
    if args.train_method == 'plain':
        print('-------------------')
        print("train method: plain")
        print('-------------------')
        result = train_model(model, data_load_dict["total_loader"], data_load_dict["test_loader"],
                             criterion, optimizer, args.epochs, device,args.num_classes,scheduler=scheduler)
        path = args.result_file_dir + "/result"

    elif args.train_method == 'DRW':
        print('-----------------')
        print("train method: DRW")
        print('-----------------')
        if args.loss_function != 'LDAM':
            raise ValueError("only LDAM can be trained with DRW")

        result = train_model_DRW(model, data_load_dict["total_loader"], data_load_dict["test_loader"],
                criterion, optimizer, args.epochs, device,args.num_classes,class_sample_num_dict,scheduler=scheduler)
        path = args.result_file_dir + f"/result"

    elif args.train_method == 'minimax':
        print('------------------------')
        print("train method: minimax_ut")
        print('------------------------')
        if args.loss_function not in minimax_loss_function_list:
            raise ValueError("only minimax type loss can be trained with minimax_ut")

        scheduler = get_minimax_schedular(optimizer, args.learning_rate)
        result = train_minimax(model, data_load_dict, criterion, optimizer, args.epochs,
                                device, args.num_classes, scheduler=scheduler)

        if args.loss_function in minimax_loss_function_list:
            path = args.result_file_dir + "/result"
        else:
            raise ValueError("train method does not match")
    else:
        raise ValueError("train method does not match")

    #save result

    save_result(result, path)
    information_list = read_result(path)
    save_result_plot(information_list, args.num_classes, path)

if __name__ == '__main__':
    main()

