import os 
import re
from pathlib import Path
import sys
import shutil
import tarfile
import time
import argparse


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Rename DL1 files.")

    parser.add_argument("--input_dir", type=str, help="the input directory")
    parser.add_argument("--tar", action="store_true", help="Enable the tar_flag (default: disabled)")

    # Parse the command-line arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    tar_flag = args.tar

    filepath = Path(input_dir)
    filepath_out = Path(os.path.dirname(__file__)).joinpath("dl1_renamed")
    current_file_pattern = f"dl1_\d+_sb_\d+_ob_\d+_\d+_\d+\..*"
    files = os.listdir(filepath)

    print("filepath:", filepath)
    print("output filepath:", filepath_out)
    if os.path.exists(filepath_out):
        shutil.rmtree(filepath_out)
    os.makedirs(filepath_out)
    for file in files:
        if re.search(current_file_pattern, file):
            fileparams=file.split(".")[0]
            fileExt=file.split(".")[1]
            fileparams=fileparams.split("_")
            threadIdx = int(fileparams[7])
            processIdx = int(threadIdx%8/2)

            print(f"Found: {file}")
            fileparams.insert(7, str(processIdx))
            outputFileName = "_".join(fileparams)
            outputFileName = ".".join([outputFileName, fileExt])
            print(outputFileName)
            shutil.copyfile(filepath.joinpath(file), filepath_out.joinpath(outputFileName))
    sb_id = fileparams[3]
    ob_id = fileparams[5]
    tarball_name = f"dl1_renamed_sb_{sb_id}_ob_{ob_id}.tar"
    tarball_path = Path(os.path.dirname(__file__)).joinpath("tarball_name")

    if tar_flag :
        print("Creating archive...")
        if os.path.exists(tarball_name):
        # If it exists, delete it
            os.remove(tarball_name)
            print(f"existing {tarball_name} has been deleted.")
        time.sleep(3)
            
        with tarfile.open(tarball_name, "w") as tar:
        # Add a file to the tar archive (you can add more files and directories)
            tar.add(filepath_out)

    if os.path.exists(filepath_out):
        shutil.rmtree(filepath_out)
    print("done")