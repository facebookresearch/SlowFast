import os
from random import randint

dataset_root = "dataset/tiny-kinetics-400/data"
classes = [dir_name for dir_name in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root,dir_name))]
train_csv = dataset_root+ "/train.csv"
val_csv = dataset_root+"/val.csv"
with open(train_csv, 'w') as f_t:
    with open(val_csv, 'w') as f_v:
        label_map = {class_name:i for i,class_name in enumerate(classes)}
        for class_ in classes[:4]:
            file_names = os.listdir(os.path.join(dataset_root,class_))
            for file_name in file_names:
                if randint(0, 100) < 80:
                    f = f_t
                else:
                    f = f_v
                f.write("{} {}\n".format(os.path.join(dataset_root,class_, file_name),label_map[class_]))
