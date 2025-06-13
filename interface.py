import gradio as gr
import numpy as np

def launch_interface(model, features):
    fields = list(features.columns)
    gradio_inputs = [gr.Number(label=field) for field in fields]

    def predict_fn(*args):
        try:
            input_dict = {field: float(val) for field, val in zip(fields, args)}
            inputs = np.array([input_dict[col] for col in fields], dtype=np.float32)
            prediction = model.predict(inputs.reshape(1, -1))
            return (np.round(prediction[0], 2),)
        except Exception as e:
            return (f"Error: {str(e)}",)

    theme = gr.themes.Citrus()

    iface = gr.Interface(
        fn=predict_fn,
        inputs=gradio_inputs,
        outputs=[gr.Textbox(label="Amina Flow")],
        title="Amina Flow Predictor",
        description="Enter process parameters to predict Amina Flow.",
        theme=theme
    )

    iface.launch(share=True)
