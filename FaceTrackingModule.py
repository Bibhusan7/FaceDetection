import cv2
import mediapipe as mp


class FaceTracking():
    def __init__(self, detectionCon = 0.5, modelSel=0):
        self.detectionCon = detectionCon
        self.modelSel = modelSel

        self.mpFaceDet = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDet = self.mpFaceDet.FaceDetection(0.75)

    def findFace(self,img):
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDet.process(imgRgb)
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxx = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxx.xmin * iw),int(bboxx.ymin * ih),int(bboxx.width * iw),int(bboxx.height * ih)
                cv2.rectangle(img, bbox, (255,0,255), 2)
                cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_TRIPLEX,
                             1,(0,255,0),1)
                l=30
                t=2
                rt=1
                x,y,w,h,=bbox
                x1,y1=x+w,y+h
                cv2.rectangle(img,bbox,(0,255,0),rt)
                #top left xy
                cv2.line(img, (x,y), (x+l,y),(255,0,255),t)
                cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
                #top right x1, y
                cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
                cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)
                #bottom left x, y1
                cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
                cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
                # bottom right x1, y1
                cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
                cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img



def main():
    cap = cv2.VideoCapture(1)
    tracker = FaceTracking()
    while True:
        success, img = cap.read()
        img = tracker.findFace(img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        cv2.imshow("Video", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()