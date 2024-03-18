from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x
    # Create a Webcam Object
    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    model = YOLO("C:/Users/gorakh/OneDrive/Desktop/Hackathon/Violence_detection_system-main/Violence_detection_system-main/best (9).pt")

    classNames = ["NONVOILENCE", "VIOLENCE"]
    
    # Define colors for each class (you can customize these colors)
    class_colors = [(0, 255, 0), (0, 255, 0)]

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'

                # Set the rectangle and label color based on the class index
                color = class_colors[cls]

                # Modern styling
                font_scale = 0.7
                text_thickness = 1
                rect_thickness = 2

                # Use a modern font and color for the labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                label_color = (255, 255, 255)  # White text color

                # Draw a modern bounding box without filling
                cv2.rectangle(img, (x1, y1), (x2, y2), color, rect_thickness)

                # Draw a modern label with a drop shadow effect
                label_size, _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                label_x = x1
                label_y = y1 - 10  # Adjust the label position
                shadow_color = (0, 0, 0)  # Black shadow color
                shadow_thickness = 2
                cv2.putText(img, label, (label_x, label_y), font, font_scale, shadow_color, shadow_thickness, cv2.LINE_AA)
                cv2.putText(img, label, (label_x, label_y), font, font_scale, label_color, text_thickness, cv2.LINE_AA)

        yield img





        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()