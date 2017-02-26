__author__ = 'YuzhongHuang'

import os
import glob
import argparse

def generate_structure(paths):
    for path in paths:
        name = path.split('/')[-1].split('_test')[0]

        out_train_full_path = os.path.join(out_train_path, name)
        out_test_full_path = os.path.join(out_test_path, name)

        try:
            os.mkdir(out_train_full_path)
        except OSError:
            pass

        try:
            os.mkdir(out_test_full_path)
        except OSError:
            pass

        train_file = open(os.path.join(out_train_full_path, 'train.txt'), 'w')
        test_file = open(os.path.join(out_test_full_path, 'test.txt'), 'w')

        with open(path) as f:
            vid_list = f.readlines()

        train_list = []
        test_list = []

        for vid in vid_list:
            vid_name = vid.split(' ')[0]
            label = vid.split(' ')[1]

            if int(label) == 1:
                train_list.append(vid_name)
            elif int(label) == 2:
                test_list.append(vid_name)

        for vid in train_list:
            train_file.write("%s\n" % vid)
        for vid in test_list:
            test_file.write("%s\n" % vid)

        train_file.close()
        test_file.close()              

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="split files")
    parser.add_argument("src_dir")
    parser.add_argument("out_dir")

    args = parser.parse_args()
    out_path = args.out_dir
    src_path = args.src_dir
    out_train_path = os.path.join(out_path, 'train/')
    out_test_path = os.path.join(out_path, 'test/')

    print out_test_path


    if not os.path.isdir(out_path):
        print "creating folder: "+out_path
        os.makedirs(out_path)

    if not os.path.isdir(out_train_path):
        print "creating folder: "+out_train_path
        os.makedirs(out_train_path)

    if not os.path.isdir(out_test_path):
        print "creating folder: "+out_test_path
        os.makedirs(out_test_path)

    vid_list = glob.glob(src_path+'/*')
    generate_structure(vid_list)
