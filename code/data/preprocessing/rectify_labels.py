import numpy as np
import os
import json

path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12.json"
dst_path = "/data1/datasets/LSU-Dataset/user_label_gautamgare_12_v1.json"

label_json_path = os.path.join(path)
with open(label_json_path, 'r') as json_file:
    label_dict = json.load(json_file)

new_label_dict = {}
for key in label_dict.keys():
    new_label_dict[f"{key.split('(')[0]}/{key}"] = label_dict[key]

#Save processed dataset dict
new_label_json = json.dumps(new_label_dict, indent=4)
f = open(dst_path, "w")
f.write(new_label_json)
f.close()
