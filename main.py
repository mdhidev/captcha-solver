#!/usr/bin/env python3

from cmath import pi
from statistics import mode
from time import sleep
from turtle import color
import uuid
import cv2
from PIL import Image
from PIL import ImageEnhance
import requests
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pytesseract
from scipy.ndimage import find_objects,label
from scipy.ndimage import measurements

def crop_image(img, tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]
  

def fast_gaussian_blur(
    img, ksize, sigma):
  kernel_1d = cv2.getGaussianKernel(ksize, sigma)
  return cv2.sepFilter2D(img, -1, kernel_1d, kernel_1d)
  
def split_images_y(img):
  imgs = []
  start = 0
  start_found = False
  end = 0
  for (idy,y) in enumerate(img[0]):
    
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
        start_found=True
        start=idy
  # print(len(imgs))
  return imgs

def split_images_x(img):
  imgs = []
  start = 0
  start_found = False
  end = 0
  for (idx,x) in enumerate(img):
    
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
        start_found=True
        start=idx
  # print(len(imgs))
  return imgs

def main():
  res = requests.get(
      'https://eduold.uk.ac.ir/Forms/AuthenticateUser/captcha.aspx')
  # id = str(uuid.uuid4())
  # Path("./dataset2/"+ id).mkdir(parents=True, exist_ok=True)
  img = Image.open(BytesIO(res.content)).convert("HSV")
  # np_img = np.array(img)
      
  # plt.subplot(2, 2, 1)
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
  #         if data[0] >= 245 or data[2] <= 60:
  #             img.putpixel((i, j), (0, 0, 255))
  
  # # img = Image.fromarray(np_img)
  plt.subplot(2, 2, 2)
  plt.imshow(img)
  # plt.show()
  # exit()
  src_img = img
  img = img.convert('L')
  # img.save("./dataset2/"+id+"/src.png")
  # plt.subplot(2, 2, 1)
  # plt.imshow(img, cmap='gray', vmin=0, vmax=255)
  # img.show()
  np_img = np.array(img)
  # np_img = [255 if pix > 163 else 0 for pix in np_img]
  np_img[45:50, 45:80] = 255
  np_img[45:50, 85:95] = 255
  np_img[45:50, 100:140] = 255
  # print("min",np.min(np_img))
  # print("max",np.max(np_img))
  # print("mean",np.mean(np_img))
  np_img = np.where(np_img > np.mean(np_img) - 5, 0, 1)

  # fig = plt.figure(figsize=(50,280))
  # plt.subplot(2, 2, 2)
  # plt.imshow(np_img, cmap='gray', vmin=0, vmax=1)
  for _ in range(2):
    for (idx,x) in enumerate(np_img):
      for (idy,y) in enumerate(x):
        if idx > 0 and idx < len(np_img) - 1 and idy > 0 and idy < len(x) - 1 and y == 1:
          tmp = np_img[
              idx - 1:idx + 2,
              idy - 1:idy + 2
          ].flatten()
          # print(idx, idy, y, np_img[
          #     idx - 1:idx + 2,
          #     idy - 1:idy + 2
          # ])
          # print(list(tmp).count(1))
          if list(tmp).count(1)-1 < 4:
            np_img[idx,idy] = 0
            
  
  # np_img=crop_image(np_img,80)
  
  # low_threshold = 50
  # high_threshold = 150
  # edges = cv2.Canny(np_img, low_threshold, high_threshold)
  
  # rho = 1  # distance resolution in pixels of the Hough grid


  # theta = np.pi / 180  # angular resolution in radians of the Hough grid
  # threshold = 15  # minimum number of votes (intersections in Hough grid cell)
  # min_line_length = 50  # minimum number of pixels making up a line
  # max_line_gap = 20  # maximum gap in pixels between connectable line segments
  # line_image = np.copy(img) * 0  # creating a blank to draw lines on

  # # Run Hough on edge detected image
  # # Output "lines" is an array containing endpoints of detected line segments
  # lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
  #                         min_line_length, max_line_gap)

  # for line in lines:
  #     for x1, y1, x2, y2 in line:
  #       cv2.line(np_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
  
  # plt.subplot(2, 2, 2)
  # plt.imshow(np_img, cmap='gray', vmin=0, vmax=1)

  # print(label(np_img))
  # print(find_objects(label(np_img.all())))
  # plt.show()
  # exit()
  
  imgs = []
  
  for img in split_images_y(np_img):
    imgs.append(split_images_x(img))
    
  for (idx,x) in enumerate(imgs):
    for (idy,y) in enumerate(x):
      if len(y) >= 3 and len(y[0]) >= 3 and list(img.flatten()).count(1) > 15:
        pil_img = Image.fromarray((y * 255).astype(np.uint8), mode="L")
        plt.subplot(2, 2, 1)
        plt.imshow(src_img, cmap='gray', vmin=0, vmax=255)
        plt.subplot(2, 2, 2)
        plt.imshow(np_img, cmap='gray', vmin=0, vmax=1)
        plt.subplot(2, 2, 3)
        plt.imshow(pil_img, cmap='gray')
        plt.show(block=False)
        img_txt = input("what char? ")
        plt.close('all')
        # img_txt = pytesseract.image_to_string(
        #     pil_img, config='--psm 10')
            #         pil_img, config='--psm ' + str(i) + ' --oem 1'))
        if img_txt != None and img_txt != "" and img_txt != " ":
          pil_img.save("./dataset3/" + img_txt[0].lower() + "/" + str(uuid.uuid4()) + ".png")

  # plt.subplot(2, 2, 3)
  # plt.imshow(imgs[1][0], cmap='gray', vmin=0, vmax=1)
  
  # plt.subplot(2, 2, 4)
  # plt.imshow(np.where(np_img == 1, 0, 255).astype('uint8'), cmap='gray')
  
  # print(np.where(np_img == 1, 0, 255))
  
  # Image.fromarray((np_img * 255).astype(np.uint8), mode="L").show()
  # np_img = (np_img * 255).astype(np.uint8)
  # # np_img = cv2.GaussianBlur(np_img, (5, 5), 0)
  # np_img = cv2.blur(np_img, (5, 5))
  # pil_img = Image.fromarray(np_img, mode="L")
  # for i in range(13):
  #   try:
  #     print(i, pytesseract.image_to_string(
  #         pil_img, config='--psm ' + str(i)))
  #   except:
  #     pass
  # pil_img.show()
  # plt.show()
  # print(np_img)
  # cv2img = cv2.cvtColor(np_img,cv2.COLOR_)
  # cv2.imshow("image", cv2img)
  # cv2.waitKey(10000)
  # cv2.destroyAllWindows()
  
for _ in range(20):  
  main()