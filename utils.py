import os
import yaml


def parse_arguments(conf_file):
    with open(conf_file, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return data


def prepare_dirs(args):
    exp_name = args["exp_name"]
    base_dir = args["local_root"]
    log_dir = os.path.join(base_dir, args["train"]["log_dir"], exp_name)
    save_dir = os.path.join(base_dir, args["train"]["save_dir"], exp_name)

    print("Log directory: ", log_dir)
    print("Save directory: ", save_dir)

    return base_dir, log_dir, save_dir

