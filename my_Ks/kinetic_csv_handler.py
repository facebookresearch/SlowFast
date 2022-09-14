import csv
from itertools import count
import os


def spliter():
    splits = ["train", "val", "test"]
    for target in splits:
        write_f = open(f"kinetics-400_{target}_reduced.csv", "w")
        wr = csv.writer(write_f)
        with open(f"kinetics-400_{target}.csv", "r") as read_f:
            rdr = csv.reader(read_f)
            lines = []
            for i, line in enumerate(rdr):
                if i % 4 == 0:
                    wr.writerow(line)

        write_f.close()


def tokenizer(line, separator) -> list:
    assert separator == "," or separator == "_", "separator should be a comma or a underbar"
    tokens = line.split(separator)
    return tokens[1] if separator == "," else tokens[0]


def remover():
    # check whether the video file name exists in csv file
    # if there is no video name, do not write that line on wirte_file
    source_dir = os.path.join("/data/hong/k400_reduced/custom/", "source")
    src_files = os.listdir(source_dir)
    unique_urls = []

    for f_name in src_files:
        curr_token = tokenizer(f_name, "_")
        if curr_token is not None:
            unique_urls.append(curr_token)

    print(f"length of unique_urls: {len(unique_urls)}, and one of them is {unique_urls[1]}")
    # return

    counter = 0
    with open(f"removed.csv", "w") as wrt_f:
        wr = csv.writer(wrt_f)
        with open(f"test.csv", "r") as rdr_f:
            rdr = csv.reader(rdr_f)
            for i, line in enumerate(rdr):
                if line[1] in unique_urls:
                    wr.writerow(line)
                else:
                    counter += 1
    print(f"missing urls in csvs: {counter}")


def csv_arranger():
    splits = ["train", "val", "test"]
    # set file descriptors and csv.writers
    write_fds = []
    for split in splits:
        write_fds.append(open(f"reduced_kinetics-400_{split}.csv", "w"))
    csv_writers = [csv.writer(cur_fd) for cur_fd in write_fds]

    with open(f"removed.csv") as read_f:
        print("removed.csv is opened")
        total_len = 33066
        # print(total_len)
        rdr = csv.reader(read_f)
        for i, line in enumerate(rdr):
            # label information at first line
            if i == 0:
                for wrt in csv_writers:
                    wrt.writerow(line)
                continue
            # test dataset
            if i >= total_len // 10 * 9:
                line[4] = "test"
                csv_writers[2].writerow(line)

            # train and validation dataset
            else:
                if i % 10 == 0:
                    line[4] = "val"
                    csv_writers[1].writerow(line)
                else:
                    line[4] = "train"
                    csv_writers[0].writerow(line)

    # close file descriptors
    for fd in write_fds:
        fd.close()


def file_cpy_arranger(split: str):
    from shutil import copy2

    source_dir = os.path.join("/data/hong/k400_reduced/custom/", "source")
    target_dir = os.path.join("/data/hong/k400_reduced/custom/", split)
    # print(os.path.exists(target_dir))

    with open(f"reduced_kinetics-400_{split}.csv", "r") as read_f:
        rdr = csv.reader(read_f)
        for i, line in enumerate(rdr):
            if i == 0:
                continue
            trgt_fname = f"{line[1]}_{int(line[2]):06}_{int(line[3]):06}.mp4"
            src_path = os.path.join(source_dir, trgt_fname)
            # print("False!!!") if not os.path.exists(src_path) else None
            copy2(src_path, target_dir)


if __name__ == "__main__":
    print("main")
    # remover()
    # csv_arranger()
    # file_cpy_arranger("train")
    # file_cpy_arranger("val")
