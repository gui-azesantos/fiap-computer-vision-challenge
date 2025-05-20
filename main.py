import cv2
from deepface import DeepFace
import pandas as pd
from tqdm import tqdm

def analisar_video(input_path, output_path, excel_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo")
        return

    # Pegando propriedades do vídeo para salvar saída com mesmo padrão
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Configurando o vídeo de saída
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    resultados = []

    print(f"Processando vídeo: {frame_count} frames, {fps:.2f} FPS, resolução {width}x{height}")

    for frame_num in tqdm(range(frame_count), desc="Processando frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Análise facial e emoção via DeepFace
        try:
            analise = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            if isinstance(analise, list):
                analise = analise[0]  # Se tiver múltiplas faces

            # Pega bounding box, emoção dominante e probabilidade
            face_region = analise["region"]
            dominant_emotion = analise["dominant_emotion"]
            emotions = analise["emotion"]

            # Desenha box e label
            x, y, w, h = face_region["x"], face_region["y"], face_region["w"], face_region["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            label = f"{dominant_emotion} ({emotions[dominant_emotion]:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Guarda dados para Excel
            resultados.append({
                "frame": frame_num,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "dominant_emotion": dominant_emotion,
                **emotions
            })

        except Exception as e:
            # Se não detectar rosto ou erro, só salvar frame sem box
            resultados.append({"frame": frame_num})

        # Salva frame anotado no vídeo de saída
        out.write(frame)

    cap.release()
    out.release()

    # Exporta para Excel (sem índice)
    df = pd.DataFrame(resultados)
    df.to_excel(excel_path, index=False)

    print(f"Análise salva em: {excel_path}")
    print(f"Vídeo anotado salvo em: {output_path}")

if __name__ == "__main__":
    input_video = "input.mp4"          
    output_video = "output_analise.mp4"
    output_excel = "analise_video.xlsx"

    analisar_video(input_video, output_video, output_excel)
