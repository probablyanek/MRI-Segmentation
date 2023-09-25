import cv2
from ultralytics import YOLO
import numpy as np
import streamlit as st

model1_path = './models/model1.pt'
model2_path = './models/model2.pt'

model1 = YOLO(model1_path)
model2 = YOLO(model2_path)



def recog(og):
    image = og.copy()
    n = 1
    while True:
        image = cv2.convertScaleAbs(image, alpha=n, beta=0)


        H, W, _ = image.shape

        results = model2(image)

        for result in results:
            if result.masks is not None:
                for j, mask in enumerate(result.masks.data):
                    mask = mask.cpu().numpy() * 255 
                    mask = cv2.resize(mask, (W, H))
                    mask = np.uint8(mask)  
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  
                    overlay = cv2.addWeighted(og, 0.6, mask, 0.4, 0) 
                    cv2.imwrite(f'./output_{j}.png', overlay) 
                    cv2.imwrite(f'./mask{j}.jpg', mask)  

                return overlay, mask
    
            else:
                print('No masks found')
                n = n*1.05


def main(img):
    res1 = model1(img)[0].probs.data.tolist()
    res2 = model2(img)[0]
    if ((res1[1]-res1[0])>0) or (res2.masks is not None):
        return recog(img)

    else:
        return 0
    

if __name__ == '__main__':

    st.title("Brain Tumor Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=0.5)


        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resp = main(img)
        # st.markdown(
        #     """
        #     <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
        #     """,
        #     unsafe_allow_html=True
        # )

        if resp != 0:
            st.markdown("<p style='font-size: 36px; color:white; font-family: Montserrat, sans-serif;font-weight: bold;'>Tumor found.</p>", unsafe_allow_html=True)

            conv_img, mask = resp
            col1, col2 = st.columns(2)

            with col1:
                if st.button('Image'):
                    st.image(conv_img, caption='Image 1', use_column_width=True)

            with col2:
                if st.button('Mask'):
                    st.image(mask, caption='Image 2', use_column_width=True)
        else:
            st.markdown("<p style='font-size: 36px; color: #4caf50; font-family: Montserrat, sans-serif;font-weight: bold;'>No tumor found.</p>", unsafe_allow_html=True)


