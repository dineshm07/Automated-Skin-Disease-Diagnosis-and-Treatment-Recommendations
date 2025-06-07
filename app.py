import gradio as gr
from gen_ai.gen_ai_model import process_image_and_generate_response

# Custom CSS
custom_css = """
body {
    font-family: 'Segoe UI', sans-serif;
    margin: 0;
    background: #f8f9fa;
}
.gradio-container {
    padding: 0;
    max-width: 100%;
    box-shadow: none;
    background: none;
}
.hero {
    text-align: center;
    padding: 60px 20px 40px;
    background: linear-gradient(135deg, #c9d6ff, #e2e2e2);
    border-bottom: 3px solid #dee2e6;
}
.hero h1 {
    font-size: 2.8em;
    font-weight: 700;
    margin-bottom: 15px;
    color: #333;
}
.hero p {
    font-size: 1.2em;
    color: #444;
    max-width: 700px;
    margin: 0 auto 25px;
}
.hero img {
    width: 180px;
    margin: 20px auto 30px;
}
.features {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    margin: 30px auto;
    padding: 0 20px;
    gap: 30px;
}
.feature-box {
    background: white;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 8px 16px rgba(0,0,0,0.05);
    width: 300px;
    text-align: center;
    transition: transform 0.3s ease;
}
.feature-box:hover {
    transform: translateY(-5px);
}
.feature-box img {
    width: 60px;
    margin-bottom: 15px;
}
.feature-box h3 {
    font-size: 1.3em;
    margin-bottom: 10px;
    color: #333;
}
.feature-box p {
    color: #555;
    font-size: 0.95em;
}
#get-started {
    background-color: #007bff;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    font-size: 1em;
    cursor: pointer;
    margin-top: 20px;
}
#get-started:hover {
    background-color: #0056b3;
}
#title {
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
    text-align: center;
}
#description {
    font-size: 1.1em;
    color: #555;
    text-align: center;
    margin-bottom: 20px;
}
#disclaimer {
    color: #B22222;
    text-align: center;
    font-size: 1.5em;
    margin-top: 20px;
}
#footer {
    text-align: center;
    margin-top: 40px;
    font-size: 0.9em;
    color: #666;
}
#footer a {
    color: #0366d6;
    text-decoration: none;
}
#footer a:hover {
    text-decoration: underline;
}
@media screen and (max-width: 768px) {
    .features {
        flex-direction: column;
        align-items: center;
    }
    .feature-box {
        width: 90%;
    }
}
"""

# Diagnose Function
def diagnose(image):
    result = process_image_and_generate_response(image)
    return (
        result["pred_class"],
        result["conf_score"],
        result["gradcam_path"],
        result["llm_output"],
        "✅ Diagnosis complete."
    )

def loading_status():
    return "⏳ Diagnosing... please wait..."

# App UI
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:

    # Landing Page
    with gr.Column(visible=True) as landing_section:
        with gr.Row(elem_classes="hero"):
            gr.HTML("""
                <div>
                    <img src="https://cdn-icons-png.flaticon.com/512/3601/3601484.png" alt="AI Icon">
                    <h1>Skin Disease Diagnosis AI</h1>
                    <p>A smart diagnostic assistant powered by deep learning and a local LLM, designed to detect and explain common skin diseases using dermoscopic images.</p>
                </div>
            """)

        with gr.Row(elem_classes="features"):
            gr.HTML("""
                <div class="feature-box">
                    <img src="https://cdn-icons-png.flaticon.com/512/2709/2709164.png" alt="Upload Icon">
                    <h3>Easy Image Upload</h3>
                    <p>Upload a dermoscopic image from your device and let the model analyze it in seconds.</p>
                </div>
            """)
            gr.HTML("""
                <div class="feature-box">
                    <img src="https://cdn-icons-png.flaticon.com/512/1091/1091863.png" alt="Model Icon">
                    <h3>AI-Powered Detection</h3>
                    <p>Uses MobileNetV2 with Grad-CAM to identify the most probable skin disease visually.</p>
                </div>
            """)
            gr.HTML("""
                <div class="feature-box">
                    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712104.png" alt="Explain Icon">
                    <h3>LLM Explanations</h3>
                    <p>Local LLM provides a short, clear explanation of the disease, causes, and remedies.</p>
                </div>
            """)
            get_started_btn = gr.Button("🚀 Get Started", elem_id="get-started")

        gr.Markdown("""
            <div id='footer'>
                💻 View source on 
                <a href="https://github.com/your-username/skin-disease-diagnosis-ai" target="_blank">GitHub</a>
            </div>
        """)

    # Diagnosis Section
    with gr.Column(visible=False) as diagnosis_section:
        gr.Markdown("<div id='title'>🧬 Skin Disease Diagnosis AI</div>")
        gr.Markdown("<div id='description'>Upload a dermoscopic image to detect and explain possible skin diseases using AI.</div>")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Upload Skin Image", type="filepath")
                submit_btn = gr.Button("🔍 Diagnose")

            with gr.Column():
                heatmap_output = gr.Image(label="Grad-CAM Heatmap")

        with gr.Row():
                with gr.Column():
                    pred_output = gr.Textbox(label="Predicted Class", interactive=False)
                
                with gr.Column():
                    conf_output = gr.Textbox(label="Confidence Score", interactive=False)

        llm_output = gr.Textbox(label="Explanation (Generated by LLM)", lines=6, interactive=False)
        status_output = gr.Markdown("")

        gr.Markdown("<div id='disclaimer'>⚠️ This tool is for educational and research purposes only. It is not intended for medical diagnosis or treatment.</div>")

        # Button Behavior
        submit_btn.click(fn=loading_status, outputs=[status_output], queue=False)
        submit_btn.click(
            fn=diagnose,
            inputs=[image_input],
            outputs=[pred_output,conf_output, heatmap_output, llm_output, status_output],
            show_progress=True
        )

        def toggle_sections():
            return gr.update(visible=False), gr.update(visible=True)

        get_started_btn.click(
            fn=toggle_sections,
            outputs=[landing_section, diagnosis_section]
        )


# Launch the app
if __name__ == "__main__":
    app.launch()





