import cv2
import mediapipe as mp
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm
import numpy as np
from collections import deque


def analyze_video_with_pose(input_path, output_path, excel_path, 
                            confidence_threshold=50, movement_threshold=25, anomaly_threshold=50, smoothing_window=5):
    # Inicialização do pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Leitura do vídeo
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
    movement_buffer = deque(maxlen=smoothing_window)

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

        # Detecção de movimento
        if pose_results.pose_landmarks:
            current_landmarks = np.array(
                [(lm.x * width, lm.y * height) for lm in pose_results.pose_landmarks.landmark]
            )

            if prev_landmarks is not None:
                deltas = np.linalg.norm(current_landmarks - prev_landmarks, axis=1)
                movement_magnitude = np.mean(deltas)
                movement_variability = np.std(deltas)

                # Suavização com média móvel
                movement_buffer.append(movement_magnitude)
                smoothed = np.mean(movement_buffer)

                if smoothed > movement_threshold:
                    movement_detected = True
                    movement_count += 1
                if smoothed > anomaly_threshold or movement_variability > anomaly_threshold:
                    anomalous_movement = True
                    anomalous_movement_count += 1

            prev_landmarks = current_landmarks
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Análise de emoção
        face_analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(face_analysis, list):
            face_analysis = face_analysis[0]
        region = face_analysis.get('region', {})

        # Se não detectar rosto, pula análise de emoções
        if not region or all(region.get(k, 0) == 0 for k in ['x','y','w','h']):
            # Apenas movimento
            anomaly = anomalous_movement
            if anomaly:
                anom_count += 1
            results.append({
                'frame': frame_number,
                'anomaly': anomaly,
                'anomaly_reason': 'Anomalous Movement' if anomalous_movement else '',
                'movement_status': 'Anomalous Movement' if anomalous_movement else 'Moving' if movement_detected else 'Static',
                'movement_magnitude': movement_magnitude,
                'movement_variability': movement_variability
            })
            output_video.write(frame)
            continue

        emotions = face_analysis.get('emotion', {})
        dom_emotion = face_analysis.get('dominant_emotion', None)
        confidence = emotions.get(dom_emotion, 0) if dom_emotion else 0

        anomaly = False
        reasons = []
        # Checa baixa confiança EMOCIONAL
        if not dom_emotion or confidence < confidence_threshold:
            if anomalous_movement:
                anomaly = True
                reasons.append('Low Confidence Emotion & Anomalous Movement')
            else:
                # Não considera anomalia só pela emoção fraca
                pass
        # Checa movimento anômalo isolado
        if anomalous_movement:
            anomaly = True
            reasons.append('Anomalous Movement')

        if anomaly:
            anom_count += 1

        # Contador de emoções válidas
        if dom_emotion and confidence >= confidence_threshold:
            emotion_counter[dom_emotion] = emotion_counter.get(dom_emotion, 0) + 1
            x,y,w,h = region['x'], region['y'], region['w'], region['h']
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
            cv2.putText(frame, f"{dom_emotion} ({confidence:.1f})", (x, max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        # Desenho do status de movimento
        mv_label = 'Anomalous Movement' if anomalous_movement else 'Moving' if movement_detected else 'Static'
        color = (0,0,255) if anomalous_movement else (0,255,0) if movement_detected else (255,0,0)
        cv2.putText(frame, mv_label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1, color, 2)

        results.append({
            'frame': frame_number,
            'x': region['x'], 'y': region['y'], 'width': region['w'], 'height': region['h'],
            'dominant_emotion': dom_emotion, 'confidence': confidence,
            'anomaly': anomaly, 'anomaly_reason': ', '.join(reasons),
            'movement_status': mv_label, 'movement_magnitude': movement_magnitude, 'movement_variability': movement_variability,
            **emotions
        })

        output_video.write(frame)

    video.release()
    output_video.release()
    pose.close()

    # Cria DataFrame
    df = pd.DataFrame(results)
    total = len(df)
    anom_pct = anom_count/total*100 if total else 0
    anom_mv_pct = anomalous_movement_count/total*100 if total else 0
    emo_pct = {emo: cnt/sum(emotion_counter.values())*100 for emo,cnt in emotion_counter.items()} if emotion_counter else {}

    # Sumário
    summary = {
        'Total Frames': [total], 'Anomalies': [anom_count], 'Anomaly %': [f"{anom_pct:.2f}%"],
        'Anomalous Movements': [anomalous_movement_count], 'Anomalous Movement %': [f"{anom_mv_pct:.2f}%"]
    }
    for emo, pct in emo_pct.items(): summary[f"{emo} %"] = [f"{pct:.2f}%"]
    summary_df = pd.DataFrame(summary)

    # Grava Excel com formatação avançada
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Detailed', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        workbook  = writer.book
        # Formatar cabeçalhos
        fmt_header = workbook.add_format({'bold': True, 'bg_color': '#DDEBF7'})
        for sheet in ['Detailed','Summary']:
            ws = writer.sheets[sheet]
            ws.set_row(0, None, fmt_header)
            # Ajusta largura de colunas
            for idx, col in enumerate(df.columns if sheet=='Detailed' else summary_df.columns):
                ws.set_column(idx, idx, max(15, len(col)+2))
        # Adiciona gráfico de pizza no Summary
        chart = workbook.add_chart({'type': 'pie'})
        # Range de dados de emoções
        emo_names = list(emo_pct.keys())
        if emo_names:
            chart.add_series({
                'name': 'Emotion Distribution',
                'categories': ['Summary', 1, len(summary_df.columns)-len(emo_names), 1, len(summary_df.columns)-1],
                'values':     ['Summary', 1, len(summary_df.columns)-len(emo_names)+len(emo_names), 1, len(summary_df.columns)-1],
            })
            ws = writer.sheets['Summary']
            ws.insert_chart('H2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

    print(f"Analysis saved to: {excel_path}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    analyze_video_with_pose('input.mp4', 'output_analysis.mp4', 'video_report.xlsx')
