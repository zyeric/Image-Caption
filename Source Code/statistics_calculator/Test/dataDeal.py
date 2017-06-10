import os
import codecs

cwd=os.getcwd()

dataFile=cwd+"/toDealData.txt"

file=codecs.open(dataFile,"r",encoding="UTF-8")


lines=[]
temp=""
for line in file:
    line=line.strip()
    t=line.split()
    if len(t)==2:
        if temp!="":
            lines.append(temp)
        temp=line
    else:
        temp+=" "+line


with codecs.open('result.txt', 'w',encoding="UTF-8") as f:
    for line in lines:
        f.write(line+"\n")