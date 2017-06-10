import json
import os
import codecs

cwd=os.getcwd()

validFile=cwd+"/result.txt"

file=codecs.open(validFile,"r",encoding="UTF-8")

cnt=0

res=[]

'''
    [
        {
            "image_id": 404464,
            "caption": "black and white photo of a man standing in front of a building"
        },
        {
            "image_id": 380932,
            "caption": "group of people are on the side of a snowy field"
        }
    ]
'''



vis=False
for line in file:
    line=line.strip()
    t=line.split(" ")
    id=int(t[0])+1
    dis=""
    flag=True
    for i in range(1,len(t)):
        if flag:
            dis=t[i]
            flag=False
        else:
            dis=dis+" "+t[i]
    temp = {'image_id': id, 'caption': dis}
    res.append(temp)

with codecs.open('result.json', 'w',encoding="UTF-8") as f:
 json.dump(res, f,ensure_ascii=False)
