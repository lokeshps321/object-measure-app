import gradio as gr
from gradio_client import Client

try:
    print("Testing client...")
    client = Client("loke007/object-measure-ai")
    res = client.predict(
        image="https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
        mode="2d",
        camera_distance_cm=30,
        api_name="/process_image"
    )
    print("Success:", type(res))
except Exception as e:
    print("Error:", e)
