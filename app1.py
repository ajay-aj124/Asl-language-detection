import cv2
import numpy as np
import pyttsx3
import time
from tensorflow.keras.models import load_model
import os

# Labels for ASL alphabet
labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z', 'space', 'nothing', 'del'
]

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def text_to_speech(text):
    engine.say(text)
    engine.runAndWait()

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    return reshaped

def main():
    model_path = r'C:\Users\vigne\PycharmProjects\bagathesh project\asl_cnn_model2.keras'

    print("Loading model...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print("Error loading model:", e)
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recognized_text = ""
    output_file = open("asl_output.txt", "w")

    last_pred_label = ""
    last_prediction_time = 0
    prediction_cooldown = 5.5  # seconds

    print("Starting real-time ASL recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        frame = cv2.flip(frame, 1)

        # Bigger bounding box
        x1, y1, x2, y2 = 50, 50, 350, 350
        roi = frame[y1:y2, x1:x2]

        processed_roi = preprocess_frame(roi)

        prediction = model.predict(processed_roi)
        class_index = np.argmax(prediction)
        confidence = prediction[0][class_index]
        pred_label = labels[class_index]

        # Draw ROI box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{pred_label} ({confidence * 100:.1f}%)"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        current_time = time.time()
        if confidence > 0.8 and (pred_label != last_pred_label) and (current_time - last_prediction_time > prediction_cooldown):
            if pred_label == 'space':
                recognized_text += ' '
            elif pred_label == 'del':
                recognized_text = recognized_text[:-1]
            elif pred_label == 'nothing':
                pass
            else:
                recognized_text += pred_label

            output_file.seek(0)
            output_file.write(recognized_text)
            output_file.truncate()
            output_file.flush()

            text_to_speech(pred_label)

            last_prediction_time = current_time
            last_pred_label = pred_label

        cv2.imshow("ASL Real-Time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    output_file.close()
    cv2.destroyAllWindows()
    os.system("notepad asl_output.txt")

if __name__ == "__main__":
    main()
