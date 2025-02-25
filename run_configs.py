import subprocess
import argparse

# List of config files
configs = []
argparser = argparse.ArgumentParser(fromfile_prefix_chars='@')
argparser.add_argument("--conf", default="", type=str, help="starting path for the configs")
argparser.add_argument("--start", default=1, type=int, help="Starting from")
args = argparser.parse_args()

#Creating a list of paths for the configs
for i in range(args.start,19):
    path = f'./{args.conf}_config/{args.conf}_{i}.txt'
    configs.append(path)

for config in configs:
    print(f"Running with config: {config}")
    subprocess.run(["python", "topo_resnet.py", f"@{config}"])
