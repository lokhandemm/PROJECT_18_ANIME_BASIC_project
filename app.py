import cv2
import numpy as np

def apply_anime_filter(frame):
    """Convert the input frame into an anime-style image."""
    
    # Step 1: Edge detection for the sketch effect
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 15)

    # Step 2: Apply bilateral filter for smooth color preservation
    color = cv2.bilateralFilter(frame, 9, 300, 300)

    # Step 3: Combine edges and color
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    anime_frame = apply_anime_filter(frame)

    # Show the output
    cv2.imshow("Anime Filter", anime_frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
