import gradio as gr
import numpy as np

def launch_interface(model_starch, model_amina, features_starch, features_amina):
    fields = []
    for col in list(features_starch.columns) + list(features_amina.columns):
        if col not in fields:
            fields.append(col)

    gradio_inputs = [gr.Number(label=field) for field in fields]

    def predict_fn(*args):
        try:
            input_dict = {field: float(val) for field, val in zip(fields, args)}
            starch_inputs = np.array([input_dict[col] for col in features_starch.columns], dtype=np.float32)
            amina_inputs = np.array([input_dict[col] for col in features_amina.columns], dtype=np.float32)
            starch_pred = model_starch.predict(starch_inputs.reshape(1, -1))
            amina_pred = model_amina.predict(amina_inputs.reshape(1, -1))
            return (np.round(starch_pred[0], 2), np.round(amina_pred[0], 2))
        except Exception as e:
            return (f"Error: {str(e)}", f"Error: {str(e)}")

    theme = gr.themes.Citrus()

    iface = gr.Interface(
        fn=predict_fn,
        inputs=gradio_inputs,
        outputs=[
            gr.Textbox(label="Starch Flow"),
            gr.Textbox(label="Amina Flow"),
        ],
        title="Froth Flotation Reagent Predictor",
        description="Enter process parameters to predict Starch and Amina Flow.",
        theme=theme
    )

    iface.launch(share=True)
