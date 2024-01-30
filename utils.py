import torch
import csv
import matplotlib.pyplot as plt

def balanced_class_rate(imbalanced_data_info):
    sample = torch.tensor(imbalanced_data_info)
    balanced_class_weight = 1/sample
    normalized_balanced_class_weight = balanced_class_weight/balanced_class_weight.sum()
    return normalized_balanced_class_weight

def make_init_class_acc_info(num_classes):
    init_class_acc_info = []
    for i in range(num_classes):
        init_class_acc_info.append([0,0])
    #init_class_acc_info[i][0] means the cumulative number of i'th class-tested which is total.
    #init_class_acc_info[i][1] means the cumulative number of i'th class-tested which is correct classified.
    return init_class_acc_info

def update_class_acc_info(class_acc_info,preds,labels):
    # type(preds) = type(labels) = torch.tensor
    for i in range(len(preds)):
        class_acc_info[labels.data[i]][0] +=1
        if labels.data[i] == preds[i]:
            class_acc_info[labels.data[i]][1] +=1
    return class_acc_info

def refine_class_acc_info(class_acc_info):
    acc_per_class = []
    for i in range(len(class_acc_info)):
        if class_acc_info[i][0] == 0:
            acc_per_class.append(0)
        else:
            acc_per_class.append(class_acc_info[i][1]/class_acc_info[i][0]*100)
    # acc_per_class[i] = accuracy for i'th class
    return acc_per_class

def refine_class_acc_info_list(class_acc_info_list):
    acc_per_class_list = []
    for i in range(len(class_acc_info_list)):
        acc_per_class = refine_class_acc_info(class_acc_info_list[i])
        acc_per_class_list.append(acc_per_class)
    # acc_per_class_list[i] = acc_per_class for i'th epoch
    return acc_per_class_list

def find_max_error_class(class_acc_info):
    acc_per_class = refine_class_acc_info(class_acc_info)
    max_error_class = 0
    for i in range(len(acc_per_class)):
        if acc_per_class[i] < acc_per_class[max_error_class]:
            max_error_class = i
    return max_error_class




def find_k_max_error_class(class_acc_info,k):
    # find k classes of the lowest accuracy
    # find k classes of the highest error
    acc_per_class = refine_class_acc_info(class_acc_info)

    if k > len(acc_per_class):
        raise ValueError("k should be not greater than the number of classes")

    max_error_class_list = [0]
    for i in range(1,len(acc_per_class)):
        index = 0
        for j in range(len(max_error_class_list)):
            if acc_per_class[max_error_class_list[j]] < acc_per_class[i]:
                index = j+1
        max_error_class_list.insert(index,i)

    k_max_error_class_list = max_error_class_list[:k]

    return k_max_error_class_list

def find_k_worst_class(acc_per_class,k):
    # find k classes of the lowest accuracy
    # find k classes of the highest error
    acc_per_class = acc_per_class

    if k > len(acc_per_class):
        raise ValueError("k should be not greater than the number of classes")

    max_error_class_list = [0]
    for i in range(1,len(acc_per_class)):
        index = 0
        for j in range(len(max_error_class_list)):
            if acc_per_class[max_error_class_list[j]] < acc_per_class[i]:
                index = j+1
        max_error_class_list.insert(index,i)

    k_max_error_class_list = max_error_class_list[:k]

    return k_max_error_class_list

def save_result(result,path):
    train_class_acc = result[0]
    test_class_acc = result[1]
    train_loss = result[2]
    train_acc = result[3]
    test_acc = result[4]

    path = path+'.csv'
    with open(path,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow([len(train_class_acc)])
        for i in range(len(train_class_acc)):
            writer.writerow(train_class_acc[i])
        for i in range(len(train_class_acc)):
            writer.writerow(test_class_acc[i])
        writer.writerow(train_loss)
        writer.writerow(train_acc)
        writer.writerow(test_acc)
        if len(result) >5:
            val_class_acc = result[5]
            pi_t_list = result[6]
            test_class_acc_at_val = result[7]
            writer.writerow([len(val_class_acc),len(pi_t_list),len(test_class_acc_at_val)])
            for i in range(len(val_class_acc)):
                writer.writerow(val_class_acc[i])
            for i in range(len(pi_t_list)):
                writer.writerow(pi_t_list[i])
            for i in range(len(test_class_acc_at_val)):
                writer.writerow(test_class_acc_at_val[i])


    f.close()

def read_result(path):
    # result_load[0][j] result of train_data at jth epoch
    # result_load[1][j] result of test_data at jth epoch
    path = path+'.csv'
    result_load = []
    with open(path,'r',newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
        #rows[0][0]: num_epoch
        #rows[1:epoch+1]: train_result
        #rows[epoch+1:2*epoch+1]: test_result
        #rows[2*epoch+1]: train_loss
        #rows[2*epoch+2]: train_acc
        #rows[2*epoch+3]: test_acc

        epoch = int(rows[0][0])

        train_result_list = []
        for i in range(1,epoch+1):
            train_result_list.append(list(map(float,rows[i])))
        result_load.append(train_result_list)

        test_result_list = []
        for i in range(epoch+1,2*epoch+1):
            test_result_list.append(list(map(float,rows[i])))
        result_load.append(test_result_list)

        result_load.append(list(map(float,rows[2*epoch+1])))
        result_load.append(list(map(float, rows[2 * epoch + 2])))
        result_load.append(list(map(float, rows[2 * epoch + 3])))

        if len(rows) > 2 * epoch + 5:
            val_result_list_num = int(rows[2 * epoch + 4][0])
            pi_t_list_num = int(rows[2 * epoch + 4][1])
            val_row_num = 2 * epoch + 5
            pi_t_list_row_num = val_row_num + val_result_list_num
            
            val_result_list = []
            for i in range(val_row_num, val_row_num+val_result_list_num):
                val_result_list.append(list(map(float, rows[i])))
            result_load.append(val_result_list)

            pi_t_list = []
            for i in range(pi_t_list_row_num, pi_t_list_row_num+pi_t_list_num):
                pi_t_list.append(list(map(float, rows[i])))
            result_load.append(pi_t_list)
            
            if len(rows) > pi_t_list_row_num+pi_t_list_num:
              test_result_at_val_list_num = int(rows[2 * epoch + 4][2])
              test_result_at_val_list_row_num = pi_t_list_row_num + pi_t_list_num
            else:
              test_result_at_val_list_num = 0
              test_result_at_val_list_row_num = 0

            test_result_at_val_list = []
            for i in range(test_result_at_val_list_row_num,
                           test_result_at_val_list_row_num+test_result_at_val_list_num):
                test_result_at_val_list.append(list(map(float, rows[i])))
            result_load.append(test_result_at_val_list)

        
    f.close()
    return result_load


def save_result_plot(result_list,num_class,save_route):
    plt.clf()
    epoch_list = [i+1 for i in range(len(result_list[0]))]

    labels = ["class {}".format(i) for i in range(num_class)]

    for i in range(num_class):
        result = result_list[0]
        fig = [j[i] for j in result]
        plt.plot(epoch_list, fig)

    plt.title("Train_data_result")
    plt.legend(labels)
    plt.xlabel('epoch')
    plt.ylabel('accuracy_per_class')
    #plt.show()
    plt.savefig(save_route+'_train_acc.png')
    plt.clf()


    for i in range(num_class):
        result = result_list[1]
        fig = [j[i] for j in result]
        plt.plot(epoch_list, fig)

    plt.title("Test_data_result")
    plt.legend(labels)
    plt.xlabel('epoch')
    plt.ylabel('accuracy_per_class')
    #plt.show()
    plt.savefig(save_route + '_test_acc.png')
    plt.clf()

    plt.plot(epoch_list, result_list[2])
    plt.title("Train_data_result")
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    #plt.show()
    plt.savefig(save_route+'_train_loss.png')
    plt.clf()

def balanced_class_rate(imbalanced_data_info):
    sample = torch.tensor(imbalanced_data_info)
    balanced_class_weight = 1/sample
    normalized_balanced_class_weight = balanced_class_weight/balanced_class_weight.sum()
    return normalized_balanced_class_weight




