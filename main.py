import sys
from streamlit import cli as stcli
import tempfile
import streamlit as st
import pandas as pd
import torch
import cv2



@st.cache()
def load_model(path='best.pt'):

    detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return detection_model


def detect_image(image, model):
    pred = model(image)
    pred_df = pred.pandas().xyxy[0].sort_values('confidence', ascending=True)
    pred_image = pred.render()[0]
    if pred_df.shape[0] > 0:
        if pred_df.confidence.iloc[0] > 0.6:
            return pred_image, pred_df.name.iloc[0]
        else:
            return pred_image, 'Авария'
    else:
        return image, 'нет аварии'




def process_video(cap, model, save=True, path_to_save='temp.mp4'):

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    stframe = st.empty()
    preds = []
    if save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path_to_save, fourcc, 25.0, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame, pred = detect_image(frame, model)
        # st.write(pred)
        if pred != 'авария' and pred != 'нет аварии':
            if save:
                out.write(frame)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            preds.append(pred)
        elif pred == 'авария':
            preds.append(pred)
    cap.release()
    if save:
        out.release()
    if len(preds) > 0:
        return pd.DataFrame(preds).reset_index().groupby(0).count().sort_values('index').index[-1]
    else:
        return 'нет аварии'


def main():
    st.title('Обработка видео на наличие аварий')

    model = load_model('best.pt')

    data_type = st.radio(
        "Выберите тип данных",
        ('обработка папки (на будущее)', 'Видео'))


    if data_type == 'Видео':
        st.header('Обработка видео')
        file = st.file_uploader('Загрузите видео')
        if file:
            st.header('Результаты распознавания')
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(file.read())
            cap = cv2.VideoCapture(tfile.name)
            pred = process_video(cap, model)
            st.text('Видео обработано')
            st.metric('Вид', pred)




if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
