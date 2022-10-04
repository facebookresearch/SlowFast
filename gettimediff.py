import os
import glob

def file_info(file_name):
    retlist = []
    with open(file_name, "r") as flog:
        while True:
            line = flog.readline()
            if not line: 
                break

            if "time_diff" in line:
                retlist.append(line)
    return retlist

def resol_n_time(input_list):
    retlist = []
    for i, unit in enumerate(input_list):
        aa = unit.split(" ")
        print(float(aa[-1][-9:-2].strip()))
        retlist.append(1)
            
    return 1

def main():
    dir = "/home/hong/slowfast/"
    image_dir = os.path.join(dir,"*.txt")
    fname_images = sorted(glob.glob(image_dir))


    fname_images = ["test128_224.txt"]
    for i_file in fname_images:
        answer = resol_n_time(file_info(i_file))
        # print(answer)


if __name__ == "__main__":
    # run main program
    main()