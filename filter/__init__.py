# add "bcm2835-v4l2" to /etc/modules

import cv2
import numpy as np
import os

rootDir = os.path.join(os.path.dirname(__file__), "assets")
faceCasDir = os.path.join(rootDir, "face.xml")
eyeCasDir = os.path.join(rootDir, "eye.xml")
noseCasDir = os.path.join(rootDir, "nose.xml")

class glasses(object):
	def __init__(self, imgName):
                self.imgDir = os.path.join(rootDir, imgName)
                
		self.face_cascade = cv2.CascadeClassifier(faceCasDir)
		self.eye_cascade = cv2.CascadeClassifier(eyeCasDir)
		self.glass_img = cv2.imread(self.imgDir)
		
		self.image = None
		self.gray = None
		self.faces = None
		self.centers = None
		self.y = None
		
		self.cap = cv2.VideoCapture(0)
		
		self.isClosed = False

	def readImage(self):
		ret, self.image = self.cap.read()
		self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

	def recFace(self):
		self.centers = []
		self.faces = self.face_cascade.detectMultiScale(self.gray, 1.3, 5)
		
		for (x, y, w, h) in self.faces:
			self.y = y
			roi_gray = self.gray[y:y + h, x:x + w]
			roi_color = self.image[y:y + h, x:x + w]
			eyes = self.eye_cascade.detectMultiScale(roi_gray)
			
			for (ex, ey, ew, eh) in eyes:
				self.centers.append((x + int(ex + 0.5 * ew), y + int(ey + 0.5 * eh)))
				
	def displayGlasses(self):
		if len(self.centers) == 2:
			glasses_width = 2.16 * abs(self.centers[1][0] - self.centers[0][0])
			overlay_img = np.ones(self.image.shape, np.uint8) * 255
			h, w = self.glass_img.shape[:2]
			scaling_factor = glasses_width / w
			
			overlay_glasses = cv2.resize(self.glass_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
			
			x = self.centers[0][0] if self.centers[0][0] < self.centers[1][0] else self.centers[1][0]
			
			x -= 0.26 * overlay_glasses.shape[1]
			self.y += 0.85 * overlay_glasses.shape[0]
			
			h, w = overlay_glasses.shape[:2]
			overlay_img[int(self.y):int(self.y + h), int(x):int(x+w)] = overlay_glasses
			
			gray_glasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
			ret, mask = cv2.threshold(gray_glasses, 110, 255, cv2.THRESH_BINARY)
			mask_inv = cv2.bitwise_not(mask)
			temp = cv2.bitwise_and(self.image, self.image, mask=mask)
			
			temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
			final_img = cv2.add(temp, temp2)
			
			cv2.imshow('Lets wear Glasses', final_img)
		else:
                        cv2.imshow('Lets wear Glasses', self.image)
			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				self.isClosed = True
	
	def close(self):
		self.cap.release()
		cv2.destroyAllWindows()

class moustache(object):
	def __init__(self, imgName):
                self.imgDir = os.path.join(rootDir, imgName)
                
		self.faceCascade = cv2.CascadeClassifier(faceCasDir)
		self.noseCascade = cv2.CascadeClassifier(noseCasDir)
		self.imgMoustache = cv2.imread(self.imgDir, -1)
		
		self.orig_mask = self.imgMoustache[:,:,3]
		self.orig_mask_inv = cv2.bitwise_not(self.orig_mask)
		
		self.imgMoustache = self.imgMoustache[:,:,0:3]
		self.origMoustacheHeight, self.origMoustacheWidth = self.imgMoustache.shape[:2]
		
		self.cap = cv2.VideoCapture(0)
		
		self.frame = None
		self.gray = None
		self.faces = []
		
		self.isClosed = False
		
	def readImage(self):
		ret, self.frame = self.cap.read()
		self.gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
	
	def recFace(self):
		self.faces = self.faceCascade.detectMultiScale(
			self.gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)
	
		for (x, y, w, h) in self.faces:
			roi_gray = self.gray[y:y+h, x:x+w]
			roi_color = self.frame[y:y+h, x:x+w]
			
			nose = self.noseCascade.detectMultiScale(roi_gray)
			
			for (nx, ny, nw, nh) in nose:
				moustacheWidth = 3 * nw
				moustacheHeight = moustacheWidth * self.origMoustacheHeight / self.origMoustacheWidth
				
				x1 = nx - (moustacheWidth / 4)
				x2 = nx + nw + (moustacheWidth / 4)
				y1 = ny + nh - (moustacheHeight / 2)
				y2 = ny + nh + (moustacheHeight / 2)
				
				if x1 < 0:
					x1 = 0
				if y1 < 0:
					y1 = 0
				if x2 > w:
					x2 = w
				if y2 > h:
					y2 = h
					
				moustacheWidth = x2 - x1
				moustacheHeight = y2 - y1
				
				moustache = cv2.resize(self.imgMoustache, (moustacheWidth, moustacheHeight), interpolation=cv2.INTER_AREA)
				mask = cv2.resize(self.orig_mask, (moustacheWidth, moustacheHeight), interpolation=cv2.INTER_AREA)
				mask_inv = cv2.resize(self.orig_mask_inv, (moustacheWidth, moustacheHeight), interpolation=cv2.INTER_AREA)
				
				roi = roi_color[y1:y2, x1:x2]
				roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
				roi_fg = cv2.bitwise_and(moustache, moustache, mask=mask)
				dst = cv2.add(roi_bg, roi_fg)
				roi_color[y1:y2, x1:x2] = dst
				
				break
	
	def displayMoustache(self):
		cv2.imshow("Video", self.frame)
		
		if (cv2.waitKey(1) & 0xFF == ord('q')):
			self.isClosed = True
			
	def close(self):
		self.cap.release()
		cv2.destroyAllWindows()
