import streamlit as st
import pandas as pd
from fastai.vision.all import PILImage, load_learner

model = load_learner("models/ball-classifier.pkl")


def main():
    st.title("Ball classification")
    st.text("This model can classify football, volleyball and tennis balls")

    file = st.file_uploader("You can upload a file", ["jpg", "jpeg", "png", "webp"])
    if not file:
        return

    img = PILImage.create(file)
    pred, pred_id, probabilities = model.predict(img)

    st.success(f"Percentage: {pred}")
    st.text(f"Probability: {probabilities[pred_id]}")

    st.image(img)

    df = pd.DataFrame(
        data={"probabilities": probabilities},
        index=["Football", "Tennis ball", "Volleyball"],
    )

    st.bar_chart(df)


main()
