from run_simulation import *


def get_minority_majority_class_info(imbalance_type,num_classes):
    minority_class = []
    majority_class = []
    if imbalance_type == 'step':
        for i in range(num_classes):
            if i < (num_classes//2):
                minority_class.append(i)
            else:
                majority_class.append(i)
    elif imbalance_type == 'LT':
        for i in range(num_classes):
            if i < int(num_classes//3*2):
                majority_class.append(i)
            else:
                minority_class.append(i)
    else:
        raise ValueError(
            'imbalance_type should be step or LT.'
        )

    return minority_class, majority_class

def get_metric(acc_per_class, minority_class_info, majority_class_info, result_list=False):
    acc_per_class_info = acc_per_class.copy()

    acc_sum = 0
    for minority_class in minority_class_info:
        acc_sum += acc_per_class[minority_class]
    minority_class_acc = acc_sum / len(minority_class_info)
    acc_sum = 0

    for majority_class in majority_class_info:
        acc_sum += acc_per_class[majority_class]
    majority_class_acc = acc_sum / len(majority_class_info)

    average_acc = sum(acc_per_class) / len(acc_per_class)

    acc_per_class_info.sort()

    worst_acc = acc_per_class_info[0]
    second_worst_acc = acc_per_class_info[1]


    #metric = [worst_acc, second_worst_acc, minority_class_acc, majority_class_acc, average_acc]

    metric = [worst_acc, average_acc]


    if result_list:
        final_test_acc = result_list[1][-1]
        worst_class_index = find_k_worst_class(acc_per_class, len(acc_per_class))
        worst_class_prior = 100 * result_list[6][-1][worst_class_index[0]]
        metric = [worst_acc, average_acc, worst_class_prior]

    return metric

def get_mean_list(metric_list,metric_standard):
    mean_list = []
    for i in range(len(metric_standard)):
        sum = 0
        for metric in metric_list:
            sum += metric[i]
        mean = sum/len(metric_list)
        mean_list.append(mean)

    return mean_list

def get_std_list(metric_list,mean_list,metric_standard):
    std_list = []
    for i in range(len(metric_standard)):
        sum = 0
        for metric in metric_list:
            sum += (metric[i]-mean_list[i])*(metric[i]-mean_list[i])
        std = (sum/len(metric_list))**(1/2)
        std_list.append(std)
    return std_list

def get_mean_std_list(metric_list,metric_standard):

    if len(metric_list) == 0:
        metric = ["-" for i in range(len(metric_standard))]
    else:
        mean_list = get_mean_list(metric_list,metric_standard)
        std_list = get_std_list(metric_list, mean_list, metric_standard)
        mean_list = np.round(mean_list,2)
        std_list = np.round(std_list, 2)
        metric = [f"{mean_list[i]}+{std_list[i]}" for i in range(len(metric_standard))] #±
    return metric

def plain_comparison(args, imbalance_ratio_list, imbalance_type_list, loss_function_list):
    #metric_standard = ['worst_acc', 'second_worst_acc', 'minority_class_acc', 'majority_class_acc', 'average_acc']
    metric_standard = ['worst accuracy', 'balanced accuracy']

    # make first line
    first_line = []
    for i in range(len(metric_standard)*2):
        first_line.append(" ")
    first_line.append(args.data_name)
    for i in range(len(metric_standard)*2):
        first_line.append(" ")

    # make second line
    second_line = ["----"]
    for imbalance_type in imbalance_type_list:
        second_line.append(imbalance_type)
        for i in range(len(metric_standard)*len(imbalance_ratio_list)-1):
            second_line.append(" ")


    # make third line
    third_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for i in range(len(metric_standard)//2):
                third_line.append(" ")
            third_line.append(ratio)
            for i in range(len(metric_standard)-len(metric_standard)//2-1):
                third_line.append(" ")

    # make fourth line
    fourth_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for metric in metric_standard:
                fourth_line.append(metric)
                

    # get result matrix

    plain_result_matrix = []
    for loss_function in loss_function_list:
        function_result = [loss_function]
        for imbalance_type in imbalance_type_list:
            for imbalance_ratio in imbalance_ratio_list:
                result_folder_name = '_'.join([args.model_name,args.data_name,loss_function,args.train_method,
                                        imbalance_type,str(imbalance_ratio),"sim_num",
                                        str(args.simulation_number)])
                result_file_dir  = os.path.join(args.save_root, result_folder_name)
                path = result_file_dir + "/result"
                if os.path.isfile(path+'.csv'):
                    result_list = read_result(path)
                    if True:
                      information_list = read_result(path)
                      #save_result_plot(information_list, len(information_list[1][-1]), path)                    

                    final_test_acc_per_class = result_list[1][-1]

                    num_class = len(final_test_acc_per_class)
                    minority_class_info, majority_class_info = get_minority_majority_class_info(imbalance_type, num_class)

                    metric = get_metric(final_test_acc_per_class, minority_class_info, majority_class_info)
                else:
                    metric = ["-" for i in range(len(metric_standard))]
                function_result.extend(metric)
        plain_result_matrix.append(function_result)

    save_name = os.path.join(args.save_root,
                             f"{args.model_name}_{args.data_name}_{args.train_method}_sim_num_{args.simulation_number}")

    with open(save_name+'.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
        writer.writerow(third_line)
        writer.writerow(fourth_line)
        for i in range(len(plain_result_matrix)):
            writer.writerow(plain_result_matrix[i])
    f.close()

def minmax_comparison(args, imbalance_ratio_list, imbalance_type_list, loss_function_list):
    #metric_standard = ['worst_acc', 'second_worst_acc', 'minority_class_acc', 'majority_class_acc', 'average_acc']
    #metric_standard = ['worst accuracy', 'balanced accuracy']
    metric_standard = ['worst accuracy', 'balanced accuracy', 'worst_class_prior']

    # make first line
    first_line = []
    for i in range(len(metric_standard) * 2):
        first_line.append(" ")
    first_line.append(args.data_name)
    for i in range(len(metric_standard) * 2):
        first_line.append(" ")

    # make second line
    second_line = ["----"]
    for imbalance_type in imbalance_type_list:
        second_line.append(imbalance_type)
        for i in range(len(metric_standard) * len(imbalance_ratio_list) - 1):
            second_line.append(" ")

    # make third line
    third_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for i in range(len(metric_standard) // 2):
                third_line.append(" ")
            third_line.append(ratio)
            for i in range(len(metric_standard) - len(metric_standard) // 2 - 1):
                third_line.append(" ")

    # make fourth line
    fourth_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for metric in metric_standard:
                fourth_line.append(metric)

    # get result matrix

    minimax_result_matrix = []
    for loss_function in loss_function_list:
        function_result = [loss_function]
        for imbalance_type in imbalance_type_list:
            for imbalance_ratio in imbalance_ratio_list:
                result_folder_name = '_'.join([args.model_name, args.data_name, loss_function, args.train_method,
                                               imbalance_type, str(imbalance_ratio), "sim_num",
                                               str(args.simulation_number)])
                result_file_dir = os.path.join(args.save_root, result_folder_name)
                path = result_file_dir + "/result"
                if os.path.isfile(path + '.csv'):
                    result_list = read_result(path)
                    if True:
                        information_list = read_result(path)
                        #save_result_plot(information_list, len(information_list[1][-1]), path)

                    final_test_acc_per_class = result_list[1][-1]

                    num_class = len(final_test_acc_per_class)
                    minority_class_info, majority_class_info = get_minority_majority_class_info(imbalance_type,
                                                                                                num_class)

                    metric = get_metric(final_test_acc_per_class, minority_class_info, majority_class_info,result_list)
                else:
                    metric = ["-" for i in range(len(metric_standard))]
                function_result.extend(metric)
        minimax_result_matrix.append(function_result)

    save_name = os.path.join(args.save_root,
                             f"{args.model_name}_{args.data_name}_{args.train_method}_sim_num_{args.simulation_number}")

    with open(save_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
        writer.writerow(third_line)
        writer.writerow(fourth_line)
        for i in range(len(minimax_result_matrix)):
            writer.writerow(minimax_result_matrix[i])
    f.close()

def plain_summary(args, imbalance_ratio_list, imbalance_type_list, loss_function_list):
    metric_standard = ['worst_acc', 'second_worst_acc', 'minority_class_acc', 'majority_class_acc', 'average_acc']
    metric_standard = ['worst accuracy', 'balanced accuracy']

    # make first line
    first_line = []
    for i in range(len(metric_standard)*2):
        first_line.append(" ")
    first_line.append(args.data_name)
    for i in range(len(metric_standard)*2):
        first_line.append(" ")

    # make second line
    second_line = ["----"]
    for imbalance_type in imbalance_type_list:
        second_line.append(imbalance_type)
        for i in range(len(metric_standard)*len(imbalance_ratio_list)-1):
            second_line.append(" ")


    # make third line
    third_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for i in range(len(metric_standard)//2):
                third_line.append(" ")
            third_line.append(ratio)
            for i in range(len(metric_standard)-len(metric_standard)//2-1):
                third_line.append(" ")

    # make fourth line
    fourth_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for metric in metric_standard:
                fourth_line.append(metric)


    # get result matrix
    sim_num_list = [1,2,3,4,5]

    plain_result_matrix = []
    for loss_function in loss_function_list:
        function_result = [loss_function]
        for imbalance_type in imbalance_type_list:
            for imbalance_ratio in imbalance_ratio_list:
                metric_list = []
                for sim_num in sim_num_list:
                    result_folder_name = '_'.join([args.model_name, args.data_name, loss_function, args.train_method,
                                                   imbalance_type, str(imbalance_ratio), "sim_num",
                                                   str(sim_num)])
                    result_file_dir = os.path.join(args.save_root, result_folder_name)
                    path = result_file_dir + "/result"
                    if os.path.isfile(path + '.csv'):
                        result_list = read_result(path)

                        final_test_acc_per_class = result_list[1][-1]

                        num_class = len(final_test_acc_per_class)
                        minority_class_info, majority_class_info = get_minority_majority_class_info(imbalance_type,
                                                                                                    num_class)

                        metric = get_metric(final_test_acc_per_class, minority_class_info, majority_class_info)
                        metric_list.append(metric)
                summary_metric = get_mean_std_list(metric_list, metric_standard)
                function_result.extend(summary_metric)
        plain_result_matrix.append(function_result)

    save_name = os.path.join(args.save_root,
                             f"{args.model_name}_{args.data_name}_{args.train_method}_summary")

    with open(save_name+'.csv','w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
        writer.writerow(third_line)
        writer.writerow(fourth_line)
        for i in range(len(plain_result_matrix)):
            writer.writerow(plain_result_matrix[i])
    f.close()


def minmax_summary(args, imbalance_ratio_list, imbalance_type_list, loss_function_list):
    #metric_standard = ['worst_acc', 'second_worst_acc', 'minority_class_acc', 'majority_class_acc', 'average_acc']
    #metric_standard = ['worst accuracy', 'balanced accuracy']
    metric_standard = ['worst accuracy', 'balanced accuracy', 'worst_class_prior']

    # make first line
    first_line = []
    for i in range(len(metric_standard) * 2):
        first_line.append(" ")
    first_line.append(args.data_name)
    for i in range(len(metric_standard) * 2):
        first_line.append(" ")

    # make second line
    second_line = ["----"]
    for imbalance_type in imbalance_type_list:
        second_line.append(imbalance_type)
        for i in range(len(metric_standard) * len(imbalance_ratio_list) - 1):
            second_line.append(" ")

    # make third line
    third_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for i in range(len(metric_standard) // 2):
                third_line.append(" ")
            third_line.append(ratio)
            for i in range(len(metric_standard) - len(metric_standard) // 2 - 1):
                third_line.append(" ")

    # make fourth line
    fourth_line = ["----"]
    for imbalance_type in imbalance_type_list:
        for ratio in imbalance_ratio_list:
            for metric in metric_standard:
                fourth_line.append(metric)

    # get result matrix
    sim_num_list = [1, 2, 3, 4, 5]

    plain_result_matrix = []
    for loss_function in loss_function_list:
        function_result = [loss_function]
        for imbalance_type in imbalance_type_list:
            for imbalance_ratio in imbalance_ratio_list:
                metric_list = []
                for sim_num in sim_num_list:
                    result_folder_name = '_'.join([args.model_name, args.data_name, loss_function, args.train_method,
                                                   imbalance_type, str(imbalance_ratio), "sim_num",
                                                   str(sim_num)])
                    result_file_dir = os.path.join(args.save_root, result_folder_name)
                    path = result_file_dir + "/result"
                    if os.path.isfile(path + '.csv'):
                        result_list = read_result(path)

                        final_test_acc_per_class = result_list[1][-1]

                        num_class = len(final_test_acc_per_class)
                        minority_class_info, majority_class_info = get_minority_majority_class_info(imbalance_type,
                                                                                                    num_class)

                        metric = get_metric(final_test_acc_per_class, minority_class_info, majority_class_info,result_list)
                        metric_list.append(metric)
                summary_metric = get_mean_std_list(metric_list, metric_standard)
                function_result.extend(summary_metric)
        plain_result_matrix.append(function_result)

    save_name = os.path.join(args.save_root,
                             f"{args.model_name}_{args.data_name}_{args.train_method}_summary")

    with open(save_name + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(first_line)
        writer.writerow(second_line)
        writer.writerow(third_line)
        writer.writerow(fourth_line)
        for i in range(len(plain_result_matrix)):
            writer.writerow(plain_result_matrix[i])
    f.close()

# terminal run example:
# python3 result_compare.py -s 1 -d CIFAR10 -a simple_data_aug --train_method plain

def compare():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', type=str, default='simulation_result')

    # simulation number
    parser.add_argument('-s', '--simulation_number', default=1, type=int,
                        help='number to indicate which simulation it is')
    parser.add_argument( '--summary', default=0, type=int,
                        help='number to indicate summary or specific simulation')


    # data_setting
    parser.add_argument('-d', '--data_name', choices =data_name_list,required=True)

    # Train setting
    parser.add_argument('--train_method', choices=train_method_list, required=True)
    parser.add_argument('--model_name', choices =model_name_list, default='Resnet32')


    args = parser.parse_args()
    
    
    if args.data_name == 'CIFAR100':
      imbalance_ratio_list = [0.2,0.1] #0.1,0.05,0.4,0.3,
    elif args.data_name == 'CIFAR10':
      imbalance_ratio_list = [0.1,0.01]
    else:
      raise ValueError("data_name does not match")

    if args.train_method == 'plain' or args.train_method == 'DRW':
        loss_function_list = plain_loss_function_list
        if args.train_method == 'DRW':
            loss_function_list = ['LDAM']

        if args.summary:
            plain_summary(args, imbalance_ratio_list, imbalance_type_list, loss_function_list)
        else:
            plain_comparison(args, imbalance_ratio_list, imbalance_type_list, loss_function_list)


    elif args.train_method == 'minimax':
        loss_function_list = minimax_loss_function_list



        if args.summary:
            minmax_summary(args, imbalance_ratio_list, imbalance_type_list, loss_function_list)
        else:
            minmax_comparison(args, imbalance_ratio_list, imbalance_type_list, loss_function_list)

    else:
        raise ValueError("train method does not match")

if __name__ == '__main__':
    compare()

