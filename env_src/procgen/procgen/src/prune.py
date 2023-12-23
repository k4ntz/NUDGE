import os
from termcolor import colored

def treat_line(line):
    fpath = " " + line
    while fpath.startswith(" "):
        fpath = fpath[1:]
    return fpath[1:-3]

new_resources = ""
for line in open("resources.cpp"):
    if line.endswith('.png",\n'):
        fpath = "../data/assets/" + treat_line(line)
        if not os.path.exists(fpath):
            print(colored(f"Found non existing {fpath}, removing", "red"))
            continue
        else:   
            print(colored(f"Found existing {fpath}", "green"))
    new_resources += line

print(new_resources)    