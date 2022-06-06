#!/usr/bin/env python3

import base64
from time import time
from PIL import Image
# from matplotlib import pyplot as plt
import requests
from io import BytesIO
import numpy as np
# from pathlib import Path
from joblib import load
# from flask import Flask, request, Response
# from flask_cors import CORS, cross_origin
import bottle
from bottle import response, request
# app = Flask(__name__)
# CORS(app)
app = bottle.app()


class EnableCors(object):
    name = 'enable_cors'
    api = 2

    def apply(self, fn, context):
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if bottle.request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)

        return _enable_cors

def split_images_y(img):
  imgs = []
  start = 0
  start_found = False
  end = 0
  for (idy, y) in enumerate(img[0]):

    tmp = img[
        :,
        idy
    ].flatten()
    if start_found:
      # print("end - start",start - end)
      if list(tmp).count(1) < 2:
        start_found = False
        end = idy
        imgs.append(img[
            :,
            start:end
        ])
    else:
      if not list(tmp).count(1) < 2:
        start_found = True
        start = idy
  # print(len(imgs))
  return imgs


def split_images_x(img):
  imgs = []
  start = 0
  start_found = False
  end = 0
  for (idx, x) in enumerate(img):

    tmp = img[
        idx,
        :
    ].flatten()
    if start_found:
      if list(tmp).count(1) < 2:
        start_found = False
        end = idx
        imgs.append(img[
            start:end,
            :
        ])
    else:
      if not list(tmp).count(1) < 2:
        start_found = True
        start = idx
  # print(len(imgs))
  return imgs


def main(img = None):
  clf = load("edu.pkl")

  id = str(time())
  if img == None:
    res = requests.get(
        'https://eduold.uk.ac.ir/Forms/AuthenticateUser/captcha.aspx')
    img = Image.open(BytesIO(res.content)).convert("HSV")
  # Path("./testset/" + id).mkdir(parents=True, exist_ok=True)
  start_time = time()
  # plt.subplot(2, 2, 2)
  # plt.imshow(img)
  # plt.subplot(2, 2, 1)                                                                                                                                                                                              
  # plt.imshow(np.array(img.convert("HSV")))
  # plt.show()
  # print(np.array(img))
  # exit()

  # plt.subplot(2, 2, 2)
  # plt.imshow(img)
  
  
  # width = img.size[0]
  # height = img.size[1]
  # for i in range(0, width):  # process all pixels
  #     for j in range(0, height):
  #         data = img.getpixel((i, j))
  #         # #print(data) #(255, 255, 255)
  #         # color = 100
  #         if (data[0] == 0 and data[1] == 0) or (data[0] >= 190):
  #             img.putpixel((i, j), (0, 0, 255))
  #         if data[0] >= 245 or data[2] <= 30:
  #             img.putpixel((i, j), (0, 0, 255))

  src_img = img.convert("RGB")
  # src_img.save("./testset/"+id+"/src.png")
  img = img.convert('L')

  np_img = np.array(img)

  np_img[46:50, 45:80] = 255
  np_img[46:50, 85:95] = 255
  np_img[46:50, 100:140] = 255
  
  np_img = np.where(np_img > np.mean(np_img) - 5, 0, 1)

  for _ in range(3):
    for (idx, x) in enumerate(np_img):
      for (idy, y) in enumerate(x):
        if idx > 0 and idx < len(np_img) - 1 and idy > 0 and idy < len(x) - 1 and y == 1:
          tmp = np_img[
              idx - 1:idx + 2,
              idy - 1:idy + 2
          ].flatten()
          if list(tmp).count(1)-1 < 4:
            np_img[idx, idy] = 0

  imgs = []

  for img in split_images_y(np_img):
    imgs.append(split_images_x(img))
    

  arr = []

  for (idx, x) in enumerate(imgs):
    for (idy, y) in enumerate(x):
        if len(y) >= 6 and len(y[0]) >= 3 and list(img.flatten()).count(1) > 15:
          pil_img = Image.fromarray(
              (y * 255).astype(np.uint8), mode="L").resize((32, 32), Image.Resampling.LANCZOS).convert("1")
          # pil_img.save('./testset/'+ id + "/" + str(idx) + "-" + str(idy) +".png")
          arr.append(np.array(pil_img).reshape((1024, -1)).flatten())
      
  # print(arr)
  prediction = ''.join(clf.predict(np.array(arr)))
  print(time()-start_time)
  print(prediction)
  # src_img.save("./testset/"+id+"/" + prediction + ".png")
  return prediction
        # prediction += clf.predict(img_flatten)
        
        
        # plt.subplot(2, 2, 1)
        # plt.imshow(src_img, cmap='gray', vmin=0, vmax=255)
        # plt.subplot(2, 2, 2)
        # plt.imshow(np_img, cmap='gray', vmin=0, vmax=1)
        # plt.subplot(2, 2, 3)
        # plt.imshow(pil_img, cmap='gray')
        # plt.show()
        # img_txt = input("what char? ")
        # plt.close('all')
        # img_txt = pytesseract.image_to_string(
        #     pil_img, config='--psm 10')
        #         pil_img, config='--psm ' + str(i) + ' --oem 1'))
        # if img_txt != None and img_txt != "" and img_txt != " ":
        #   pil_img.save("./dataset3/" +
        #                img_txt[0].lower() + "/" + str(uuid.uuid4()) + ".png")

@app.route('/edu',method=['POST'])
def edu():
    try:
      r=request
      base64data = r.forms.get('img')
      # print(base64data)
      img = Image.open(BytesIO(base64.b64decode(base64data))).resize((140,50))
      return {"captcha": main(img),  "status": "OK"}
    except Exception as e:
      return {"status": "ERROR", "message": str(e)}


app.install(EnableCors())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port="8000",debug=True)
# main()