import cv2
import mediapipe as mp


#init
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawin utilities

#detection function
def detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting color from bgr to rgb
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # converting color from rgb to bgr
    return image,results


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    cv2.imshow('Video Feed', frame)
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()