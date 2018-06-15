import numpy as np

import matplotlib.pyplot as plt
import math
import cv2
import sys



def main():
	img=cv2.imread(sys.argv[1])
	rows, cols, channels = img.shape
	row4=int(rows/2)
	col4=int(cols/2)
	labels=np.zeros((2,2))

	for i in range(2):
		for j in range(2):
			new = img[i*row4:(i+1)*row4, j*col4:(j+1)*col4]
			cv2.imwrite("single"+str(i)+str(j)+".jpg", new)
			label=str(i)+str(j)
			labels[i][j]=label

	print (labels)

main()