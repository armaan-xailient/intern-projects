from xailient import dnn
import cv2 as cv
from os import listdir
from os.path import isfile, join
#By default Low resolution DNN for face detector will be loaded.
#To load the high resolution Face detector please comment the below lines.
detectum = dnn.Detector()
THRESHOLD = 0.4 # Value between 0 and 1 for confidence score


mypath = '../data/'
onlyfiles = [f for f in listdir(mypath)]
for image_file in onlyfiles:
	if 'output' not in image_file:

		im = cv.imread(f'../data/{image_file}')
		_, bboxes = detectum.process_frame(im, THRESHOLD)

		# Loop through list (if empty this will be skipped) and overlay green bboxes
		for i in bboxes:
			cv.rectangle(im, (i[0], i[1]), (i[2], i[3]), (0, 255, 0), 3)
		
		output_file = "test_output"
		if ".jpg" in image_file:
			output_file = image_file[0:-4] + "_output.jpg"
		if ".jpeg" in image_file:
			output_file = image_file[0:-5] + "_output.jpeg"
		if ".png" in image_file:
			output_file = image_file[0:-4] + "_output.png"

		cv.imwrite(f'../data/{output_file}', im)
	else:
		pass







