import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import cvzone
import math
from ultralytics import YOLO

# Function to send email
def send_email(subject, message, to_email):
    # Set up SMTP server
    smtp_server = 'smtp.gmail.com'
    port = 587  # Gmail SMTP port
    sender_email = 'kavincse943@gmail.com'
    password = 'lcae cete ekpk msds'

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Connect to SMTP server and send email
    with smtplib.SMTP(smtp_server, port) as server:
        server.starttls()  #TLS encryption(transport layer security)
        server.login(sender_email, password)
        server.send_message(msg)

# Initialize webcam
cap = cv2.VideoCapture(0)  # for webcam
cap.set(3, 640)
cap.set(4, 480)

# Load YOLO model
model = YOLO('sign.pt')
classNames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z']

text = ""  # Initialize an empty string to store detected letters

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # confidence
            conf = math.ceil((box.conf[0]) * 100) / 100

            # class Name
            cls = int(box.cls[0])
            letter = classNames[cls]
            cvzone.putTextRect(img, f'{letter}', (max(0, x1), max(20, y1)), scale=1, thickness=1, offset=5)

            text += letter  # Append detected letter to the text string

    cv2.imshow("Results", img)
    key = cv2.waitKey(1)

    # Check if the pressed key is a dot ('.')
    if key == ord('.'):
        print("Detected Text:", text)
        send_email("Detected Text", text, "kavinramasamy2003@gmail.com")  # Send email with detected text
        text = ""  # Reset text string for new input

    # Check if the pressed key is the Esc key
    if key == 27:
        break  # Break out of the loop if Esc key is pressed

cap.release()
cv2.destroyAllWindows()