import cv2
import mediapipe as mp
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import numpy as np


def analyze_video_with_pose(input_path, output_path, excel_path, confidence_threshold=20, movement_threshold=15, anomaly_threshold=30):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        print("Error: Failed to open the video.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = []
    anom_count = 0
    emotion_counter = {}
    movement_count = 0
    anomalous_movement_count = 0

    prev_landmarks = None

    print(f"Video info: {total_frames} frames | {fps:.2f} FPS | Resolution: {width}x{height}")

    for frame_number in tqdm(range(total_frames), desc="Analyzing frames"):
        ret, frame = video.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        movement_detected = False
        anomalous_movement = False
        movement_magnitude = 0
        movement_variability = 0

        if pose_results.pose_landmarks:
            current_landmarks = np.array(
                [(lm.x * width, lm.y * height) for lm in pose_results.pose_landmarks.landmark]
            )

            if prev_landmarks is not None:
                deltas = np.linalg.norm(current_landmarks - prev_landmarks, axis=1)
                movement_magnitude = np.mean(deltas)
                movement_variability = np.std(deltas)

                if movement_magnitude > movement_threshold:
                    movement_detected = True
                    movement_count += 1

                if movement_magnitude > anomaly_threshold or movement_variability > anomaly_threshold:
                    anomalous_movement = True
                    anomalous_movement_count += 1

            prev_landmarks = current_landmarks
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Emotion Analysis
        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]

            region = analysis.get("region", {})
            dominant_emotion = analysis.get("dominant_emotion", None)
            emotions = analysis.get("emotion", {})

            if dominant_emotion:
                confidence = emotions.get(dominant_emotion, 0)
            else:
                confidence = 0

            if not dominant_emotion or confidence < confidence_threshold:
                anom_count += 1
                results.append({
                    "frame": frame_number,
                    "anomaly": True,
                    "movement_status": "Anomalous Movement" if anomalous_movement else "Static" if not movement_detected else "Moving",
                    "movement_magnitude": movement_magnitude,
                    "movement_variability": movement_variability
                })
                continue

            emotion_counter[dominant_emotion] = emotion_counter.get(dominant_emotion, 0) + 1

            x, y, w, h = region.get("x", 0), region.get("y", 0), region.get("w", 0), region.get("h", 0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{dominant_emotion} ({confidence:.2f})"
            cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Movement status
            if anomalous_movement:
                movement_label = "Anomalous Movement"
                color = (0, 0, 255)
            elif movement_detected:
                movement_label = "Moving"
                color = (0, 255, 0)
            else:
                movement_label = "Static"
                color = (255, 0, 0)

            cv2.putText(frame, movement_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            results.append({
                "frame": frame_number,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "anomaly": False,
                "movement_status": movement_label,
                "movement_magnitude": movement_magnitude,
                "movement_variability": movement_variability,
                **emotions
            })

        except Exception as e:
            anom_count += 1
            results.append({
                "frame": frame_number,
                "anomaly": True,
                "movement_status": "Anomalous Movement" if anomalous_movement else "Static" if not movement_detected else "Moving",
                "movement_magnitude": movement_magnitude,
                "movement_variability": movement_variability
            })
            continue

        output_video.write(frame)

    video.release()
    output_video.release()
    pose.close()

    df = pd.DataFrame(results)

    total_frames_analyzed = len(df)
    anom_percent = (anom_count / total_frames_analyzed) * 100
    anomalous_movement_percent = (anomalous_movement_count / total_frames_analyzed) * 100

    emotion_percentages = {k: (v / sum(emotion_counter.values())) * 100 for k, v in emotion_counter.items()} if emotion_counter else {}

    summary_data = {
        "Total Frames": [total_frames_analyzed],
        "Anomalies": [anom_count],
        "Anomaly %": [f"{anom_percent:.2f}%"],
        "Anomalous Movements": [anomalous_movement_count],
        "Anomalous Movement %": [f"{anomalous_movement_percent:.2f}%"]
    }

    for emo, perc in emotion_percentages.items():
        summary_data[f"{emo} %"] = [f"{perc:.2f}%"]

    summary_df = pd.DataFrame(summary_data)

    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="Detailed Results", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"Analysis saved to: {excel_path}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    input_video_path = "input.mp4"
    annotated_video_path = "output_analysis.mp4"
    excel_output_path = "video_analysis.xlsx"

    analyze_video_with_pose(input_video_path, annotated_video_path, excel_output_path)
