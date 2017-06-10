import json
import os
import codecs

cwd=os.getcwd()

validFile=cwd+"/valid.txt"

file=codecs.open(validFile,"r",encoding="UTF-8")

cnt=0

res={"annotations":[],"images":[],"type":"captions"}

'''
    dict = {
        "annotations": [{'id': 37, 'caption': 'A bicycle replica with a clock as the front wheel.', 'image_id': 203564},
                        {'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.','image_id': 179765}
                        ]
    }
'''

number=0
res["annotations"]=[]
print(res["annotations"])
for line in file:
    line=line.strip()
    if line.isdigit():
        number=int(line)
    else:
        flag=True
        newline=""
        for word in line:
            if flag:
                newline=word
                flag=False
            else:
                newline=newline+" "+word
        temp={'id': cnt, 'caption':newline, 'image_id': number}
        res["annotations"].append(temp)
        temp = {'id': number }
        res["images"].append(temp)
    cnt=cnt+1


res["type"]="captions"
print(res["annotations"])

with codecs.open('valid.json', 'w',encoding="UTF-8") as f:
 json.dump(res, f,ensure_ascii=False)

print res['type']