import cv2

cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

def detect_cats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=6, minSize=(30, 30))
    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Unable to capture video")
        break 
    output = detect_cats(frame)
    cv2.imshow('Cat Detector', output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()