## conver the original hotpot json file to csv file, then we train the paragraph selection model

import csv
import json
import logging
from typing import Any, Dict, List, Tuple
import gzip , copy, random
file_input = "/Users/sf/CogQA/examples/hotpot_train_v1.1_500_refined.example.json"
file_output = "/Users/sf/CogQA/examples/hotpot_train_v1.1_500_refined.example.csv"
# file_input = "/Users/sf/DFGN-pytorch/hotpot_dev_distractor_v1.json"
# file_input = "/home/sudan/ACL2020/data/hotpot/hotpot_train_v1.1.json"
# file_output = "/home/sudan/ACL2020/data/hotpot/hotpot_train_v1.1.csv"
with open(file_input, "r", encoding='utf-8') as reader:
        data = json.load(reader)

csvfile = open(file_output, 'w')
writer = csv.writer(csvfile)
for line1 in data:
    line1['title'] = ""  # we add this column
    line1['label'] = ""
    keys=line1.keys()
    print(keys)
    writer.writerow(keys)
    break

for line2 in data:
    context_list = line2['context']
    supporting_list = line2['supporting_facts']
    supporting_title_list = []
    for s in supporting_list:
        supporting_title_list.append(s[0])  # the supporting title
    
#     print(len(context_list))
    line = line2
    line['context'] = []
    line['title'] = ""
    line['label'] = 0
    
    for c in context_list:
        line['context'] = c[1] # the sentences
        line['title'] = c[0] # the title
        if(line['title'] in supporting_title_list):
            line['label'] = 1
            print(supporting_title_list)
            print(line['title'])
        
#         print(line)
#         print("hello world")
        writer.writerow(line.values())
    

csvfile.close()