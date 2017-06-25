"""
The service to return the classes in the image 
You can specify the port number to use on the command line like this if you can't or don't want to use the default: python image_processing_service.py 80
"""

import timeit
import numpy as np
import urllib
import json
import os
import sys
import web # if this is not installed, it can be installed by command: easy_install web.py
import math
import random
from urllib2 import URLError

#load json file
with open('config.json') as data_file:
    config = json.load(data_file)

#load caffe
caffe_root = config["configuration"]["caffe_root"]

caffe_python = os.path.join(caffe_root, "python")
sys.path.insert(0, caffe_python)
import caffe


#load caffe model
caffe_proto = config["configuration"]["model_prototxt"]
caffe_model = config["configuration"]["model_weights"]
caffe_mean = config["configuration"]["model_mean"]
use_gpu = config["configuration"]["use_gpu"]
batch_max = config["configuration"]["batch_max"]
input_width = config["configuration"]["img_width"]
input_height = config["configuration"]["img_height"]
tmp_dir = config["configuration"]["tmp_dir"]
class_names_file = config["configuration"]["class_names"]
output_layer_name = config["configuration"]["output_layer_name"]

def loadClassNames(classNms):
    file = open(classNms, "r")
    parsedJson=json.load(file)
    return parsedJson["classes"]

class_names = loadClassNames(class_names_file)


#load net and return it
def loadnet(proto,model,use_gpu):
    if use_gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()

    net = caffe.Net(str(proto), str(model), caffe.TEST)
    return net

def readmean(meanfile):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(meanfile, 'rb').read()
    blob.ParseFromString(data)
    arr = np.array(caffe.io.blobproto_to_array(blob))
    out = arr[0]
    return out

#load net and mean file before starting the web service
net = loadnet(caffe_proto, caffe_model, use_gpu)
meanVals = readmean(caffe_mean)

#webservice configs:
urls = (
    '/', 'feed_forward'
)
app = web.application(urls, globals())


class feed_forward:
    def GET(self):
        params = web.input(json='',confMinInt=-1,confMaxNoCls=sys.maxint)
        jsonfile = params['json']
        #if json is not given, load it from a file on the server (I did it for testing)
        if jsonfile=='':
            jsonfile = 'http://localhost:8080/static/json/jsonfile.json'
            jsonfile= urllib.urlopen(jsonfile)
            jsonfile = jsonfile.read()
        confMinInt = float(params['confMinInt'])
        confMaxNoCls = int(params['confMaxNoCls'])
        return self.predict(jsonfile,confMinInt,confMaxNoCls)

    def POST(self):
        params = web.input(json='', confMinInt=-1, confMaxNoCls=sys.maxint)
        jsonfile = params['json']
        if jsonfile == '':
            jsonfile = 'http://localhost:8080/static/json/jsonfile.json'
            jsonfile = urllib.urlopen(jsonfile)
            jsonfile = jsonfile.read()
        confMinInt = float(params['confMinInt'])
        confMaxNoCls = int(params['confMaxNoCls'])
        return self.predict(jsonfile, confMinInt, confMaxNoCls)

    #run network's forward and compute
    def predict(self,jsonfile , confMinInt , confMaxNoCls):
        start_time = timeit.default_timer()
        try:
            output_json = {}
            output_json['results'] = []

            #if you want to pass the json file address, you can uncomment the line below
            # response = urllib.urlopen(jsonfile)
            # inputjson = json.loads(response.read())

            #if the json is passed as text, uncomment the line below:
            inputjson = json.loads(jsonfile)
            web.header('Content-Type', 'application/json')

            imgUrls = inputjson["images"]
            imgs = self.downloadImgs(imgUrls, tmp_dir)
            noForwards = int(math.ceil(float(len(imgs)) / float(batch_max)))
            for i in range(noForwards):
                noCurItBatch = min(len(imgs), (i + 1) * batch_max)
                self.injectImgData(net, imgs[i * batch_max: noCurItBatch], batch_max, meanVals, input_height, input_width)
                net.forward()
                for j in range(noCurItBatch - i * batch_max):
                    imgClsIds, imgClsConfs = self.getClassConfs(net, output_layer_name, j, confMinInt, confMaxNoCls)
                    self.addToJSonOutput(class_names, output_json, imgUrls[i * batch_max + j], imgClsIds, imgClsConfs)

            elapsed = timeit.default_timer() - start_time
            output_json = json.dumps(output_json)
            print 'executation time: ' + str(elapsed)
            self.emptyTmp(tmp_dir)
            return output_json
        except TypeError as err:
            return self.generateErrorJson(str(err))
        except IOError as err:
            return self.generateErrorJson(str(err))
        except OSError as err:
            return self.generateErrorJson(str(err))
        except ValueError as err:
            return self.generateErrorJson(str(err))
        except URLError as err:
            return self.generateErrorJson(str(err))
        except:
            return self.generateErrorJson('unexpected error: {0}'.format(sys.exc_info()[0]))
    #generate json with the code for exception
    def generateErrorJson(self,errDesc):
        output_json = {"error" : errDesc}
        return json.dumps(output_json)

    def getClassConfs(self,net,outputLrNm,batchNo,confAllowedMin,allowedNoClsMax):
        #get the last layer of the network and analyze it
        imgClsIds=[]
        imgClsConfs=[]
        #get the last blob, note that this might not always be safe, maybe its better to get the output name from user?
        # keys=net.blobs.keys()
        # data = net.blobs[keys[len(keys)-1]].data
        data = net.blobs[outputLrNm].data[batchNo]
        #analyze data
        sortedIndices = sorted(range(len(data)), key=lambda x: data[x])
        #revert the indices and get the (allowedNoClsMax) last elements if required
        if len(sortedIndices)> allowedNoClsMax:
            sortedIndices = sortedIndices[:-(allowedNoClsMax+1):-1]
        else:
            sortedIndices = sortedIndices[::-1]

        sortedCnfs = data[sortedIndices]

        #remove confidences which are smaller than the value given
        idsOfSortedIds = np.where(sortedCnfs>=confAllowedMin)[0]
        sortedIndices = map(sortedIndices.__getitem__, idsOfSortedIds)
        sortedCnfs = map(sortedCnfs.__getitem__, idsOfSortedIds)
        return (sortedIndices,sortedCnfs)


    #add the outputs corresponding to the image to the output json
    def addToJSonOutput(self,clsNames,jsonObj,imgUrl,imgClassIds,imgClassConfs):
        imgDic = {"url":imgUrl}
        imgclasses=[]

        for clId , confd in zip(imgClassIds,imgClassConfs):
            imgClassesDic = {'class': clsNames[clId] , "confidence": float(confd) }
            imgclasses.append(imgClassesDic)

        imgDic.update({'classes': imgclasses })
        jsonObj["results"].append(imgDic)

    # empty tmp directory
    def emptyTmp(self,tmpDir):
        map(os.unlink, (os.path.join(tmpDir, f) for f in os.listdir(tmpDir)))

    # downloads imgs into the tmp dir
    def downloadImgs(self,imgUrls, tmpDir):
        outputImgs = []
        for img in imgUrls:
            imgName = os.path.basename(img)
            outputImgName = str(random.random()).replace('.','') + imgName.replace(' ','_')
            outputAddr = os.path.join(tmpDir, outputImgName)
            urllib.urlretrieve(img, outputAddr)
            outputImgs.append(outputAddr)
        return outputImgs

    # insert images as input to the network
    def injectImgData(self,input_net, imgs, batchmax, meanvals, imgW, imgH):
        transformer = caffe.io.Transformer({'data': input_net.blobs['data'].data.shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', meanvals)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))
        batchsize = min(len(imgs), batchmax)
        input_net.blobs['data'].reshape(batchsize, 3, imgH, imgW)
        for i in range(batchsize):
            input_net.blobs['data'].data[i] = transformer.preprocess('data', caffe.io.load_image(imgs[i]))
        input_net.reshape()


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()


