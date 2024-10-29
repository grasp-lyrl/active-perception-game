import json
import csv
from pprint import pprint

# scene = "102344280"
# scene = "102344529"
# scene = "102344250"
scene = "102816036"
file = open(
    "../data/scene_datasets/hssd-hab/scenes/" + scene + ".scene_instance.json", "r"
)

obj_data = json.load(file)

# print(obj_data['object_instances'][0].keys())

# read csv
csv_filepath = "./objects.csv"
csv_file = open(csv_filepath, "r")
csv_data = csv.reader(csv_file, delimiter=",")
obj_names = {}
for row in csv_data:
    obj_names[row[0]] = row[14]

obj_ids = []
for obj in obj_data["object_instances"]:
    obj_ids.append(obj["template_name"])

# print(obj_data['object_instances'][0]['template_name'])
# name = obj_data['object_instances'][0]['template_name']
# print(obj_names[name])

# for id in obj_ids:
#     print(id, obj_names[id])

# need to map template id --> object name --> object lbael

# read obj classes json
obj_classes_filepath = "./hssd-hab_semantic_lexicon.json"
obj_classes_data = json.load(open(obj_classes_filepath, "r"))["classes"]

obj_classes = {}  # name: label
for ocd in obj_classes_data:
    obj_classes[ocd["name"]] = ocd["id"]

valid_obj_ids = {}  # id: name
for id in obj_ids:
    if id in obj_names.keys() and obj_names[id] != "":
        valid_obj_ids[id] = obj_names[id]

valid_labeled_obj_ids = {}  # id: label
for id in valid_obj_ids:
    if valid_obj_ids[id] in obj_classes.keys():
        valid_labeled_obj_ids[id] = obj_classes[valid_obj_ids[id]]


# print(valid_labeled_obj_ids)

# store id: label, name, location, rotation
objects_final = {}
for obj in obj_data["object_instances"]:
    if obj["template_name"] in valid_labeled_obj_ids.keys():
        of = {}
        of["id"] = obj["template_name"]
        of["label"] = valid_labeled_obj_ids[obj["template_name"]] - 1
        of["name"] = valid_obj_ids[obj["template_name"]]
        of["location"] = obj["translation"]
        of["rotation"] = obj["rotation"]
        objects_final[obj["template_name"]] = of

# save dict
with open("objects_" + scene + ".json", "w") as fp:
    fp.write(json.dumps(objects_final))

# # read gt objects file
# gt_obj_filepath = './objects_102344280.json'
# gt_obj_json = json.load(open(gt_obj_filepath))

# # get num gt objs and locations
# gt_obj_locs = {i:[] for i in range(28)}
# gt_objs_num = {i:0 for i in range(28)}
# for tid, obj in gt_obj_json.items():
#     gt_objs_num[obj['label']] += 1
#     gt_obj_locs[obj['label']].append(obj['location'])
# print(gt_objs_num)

# for i in range(28):
#     print('Semantic class {}: {} objects'.format(i+1, gt_objs_num[i]))
#     # print locations
#     pprint(gt_obj_locs[i])
