import cv2
import numpy as np
import imutils
import requests
import time

last_detection_time = 0
def send_notification_with_image(image_path, message):
    # Token you received from notify-bot line
    token = 'fzWAjzY3KfFHcysUg6g5jeQ4YYmdIec15XYRtkcK2GP'  
    url = 'https://notify-api.line.me/api/notify'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    # Create payload and files to send
    payload = {'message': message}
    files = {'imageFile': open(image_path, 'rb')}

    response = requests.post(url, headers=headers, data=payload, files=files)
    print(response.text)

# Set class and color
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "GUN", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "PERSON", "MOTORBIKE", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
COLORS = np.array([[50, 205, 50]] * len(CLASSES)) # Set color green [50, 205, 50]

gun_cascade = cv2.CascadeClassifier('D:/opencv/Model/cascade.xml') # Load Cascade Classifier for guns

def main(): 
    # Load object detection model from file
    net = cv2.dnn.readNetFromCaffe("D:/opencv/Model/MobileNetSSD.prototxt","D:/opencv/Model/MobileNetSSD.caffemodel") 
    global last_detection_time
    last_detection_time = 0

    while True:
        print("Select mode:")
        print("1. VDO")
        print("2. Camera")
        print("3. Exit")
        mode = input("Enter mode (1/2/3): ")

        if mode == '1':
            video_path = 'D:/opencv/Date/gun.mp4' #address of video
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = imutils.resize(frame, width=600)
                processed_frame = process_frame(frame, net) #Call function process_frame
                # Display processed_frame results obtained from process_frame
                if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    cv2.imshow('Output', frame) 
                else:
                    print("Invalid frame")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                
        elif mode == '2':
        # Camera mode
            cap = cv2.VideoCapture(0) # Open the camera
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = process_frame(frame, net) #Call function process_frame
                cv2.imshow('Output', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break

        elif mode == '3':
            # Exit
            break
        else:
            print("Invalid mode. Please Select 1, 2, or 3.")

        # Clear the video and close all OpenCV windows.
        cap.release()
        cv2.destroyAllWindows()
    
def process_frame(frame, net):
        global last_detection_time
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # detect Object
        for i in np.arange(0, detections.shape[2]):
            percent = detections[0, 0, i, 2]
            if percent > 0.5:
                class_index = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (50, 205, 50), 2)
                cv2.rectangle(frame, (startX - 1, startY - 30), (endX + 1, startY), (50, 205, 50), cv2.FILLED)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        # detect GUN
        guns = gun_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        for (x, y, w, h) in guns:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "GUN", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Added a condition that checks if a gun is found or not.
        if len(guns) > 0:
            current_time = time.time()
            if current_time - last_detection_time >= 10 or last_detection_time == 0:
                last_detection_time = current_time
                for i, (x, y, w, h) in enumerate(guns):
                    cv2.imwrite('detected_gun.jpg', frame)
                    send_notification_with_image('detected_gun.jpg', 'Dangerous detects gun!!!') # Enter the message you want to send to. notify-botline

if __name__ == "__main__":
    main()
