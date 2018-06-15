import cv2
import sys
from os import listdir



def main():
	hazard=str(sys.argv[1])
	folder='test/'+hazard+'/'
	files=[f for f in listdir(folder)]
	for i in files:
		img=cv2.imread(folder+i)
		rows, cols, channels = img.shape
		for j in range(3):
			M = cv2.getRotationMatrix2D((cols/2,rows/2),120*j,1)
			dst = cv2.warpAffine(img,M,(cols,rows))
			cv2.imwrite(folder+i[:-4]+str(j)+'.jpg', dst)
	#print(files)

main()
