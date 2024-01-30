import os, csv

def result_folder_maker(args):
    args.result_dir = os.path.join(args.save_root)
    args.result_file_dir = os.path.join(args.save_root, args.result_folder_name)

    folder_list = [args.save_root,args.result_dir,args.result_file_dir]

    for folder in folder_list:
        if not os.path.exists(folder):
            os.mkdir(folder)
