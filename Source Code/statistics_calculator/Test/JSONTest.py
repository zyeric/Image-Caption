import json

dataDir='..'
dataType='val2014'
algName = 'fakecap'
annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)

f=open(annFile)

setting=json.load(f)

# annotations
# type
# licenses
# info
# images
for s in setting:
    print(s)


print("type:",setting["type"])
print("licenses:",setting["licenses"])
print("info:",setting["info"])
'''
annotations(image_id)  ===   images(id)
'''
for i in range(100):
    print("annotations:",setting['annotations'][i])

for i in range(20):
    print("images:",setting["images"][i])

for i in range(len(setting['annotations'])):
    annotation=setting['annotations'][i]
    if(annotation["image_id"]==391895):
        print(i,".................",annotation)



