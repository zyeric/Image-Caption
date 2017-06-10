from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and pathes
dataDir='.'
dataType='valid'
algName = 'result'
annFile='%s/Test/%s.json'%(dataDir,dataType)
subtypes=['result', 'evalImgs', 'eval']
[resFile, evalImgsFile, evalFile]= \
['%s/Test/%s.json'%(dataDir,algName) for subtype in subtypes]

# create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# cocoEval.params['image_id'] = cocoRes.getImgIds()
# please remove this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()

print('final result:')

for metric, score in cocoEval.eval.items():
    print('%s: %.3f'%(metric, score))

