import subprocess
import argparse
import os

# List of config files
# simple add to send it to git KEKW
configs = []
argparser = argparse.ArgumentParser(fromfile_prefix_chars='@')
argparser.add_argument("--conf", default="", type=str, help="starting path for the configs")
argparser.add_argument("--start", default=0, type=int, help="Starting from")
argparser.add_argument("--dir", default="", type=str, help="Giving the directory for the configs to run")
argparser.add_argument("--file", default="", type=str, help="Giving the file to run")
args = argparser.parse_args()

def get_file_list(directory):
  """Returns a list of files in the specified directory."""
  file_list = []
  for entry in os.listdir(directory):
    full_path = os.path.join(directory, entry)
    if os.path.isfile(full_path):
      file_list.append(entry)
  return file_list

if args.file != '':
    subprocess.run(["python", "main.py", f"@{args.file}"])

if args.dir == "":
    #Creating a list of paths for the configs
    for i in range(args.start,19):
        path = f'@{args.conf}'
        configs.append(path)

else:
   files = get_file_list(args.dir)
   i = args.start
   for j in range(i,len(files)):
      configs.append(args.dir + files[j])
   print(files)
for i,config in enumerate(configs):
    print(f"Running with config: {config}, id: {i+args.start}")
    subprocess.run(["python", "main.py", f"@{config}"])
