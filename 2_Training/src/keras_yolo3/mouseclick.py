import time
import cv2
mousePoints = list()
    

def video_click(videopath):
    capture_time = 15
    cap = cv2.VideoCapture(videopath)
    start_time = time.time()
    while ( int(time.time() - start_time) < capture_time ):
        ret,frame = cap.read()
        
        def mouseClick(event,x,y,flags,param):
                global mousePoints
                if event==cv2.EVENT_LBUTTONDOWN:
                    mousePoints.append((x,y))
                '''if event==cv2.EVENT_LBUTTONUP:
                    mousePoints.append((x,y))'''
                    
        cv2.setMouseCallback('result',mouseClick)
        # print(mousePoints)
        # cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("result",600,600)
        cv2.imshow('result',frame)
        cv2.waitKey(27)
        
    cap.release()
    cv2.destroyAllWindows()
    return mousePoints

