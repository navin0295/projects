# Execution Instructions

## Backend — Google Colab
1. Open `hi_Fin_Chatbot (1).ipynb` in Google Colab.
2. In the cell containing `!ngrok authtoken ""`, set your ngrok token:

```bash
!ngrok authtoken <YOUR_NGROK_AUTH_TOKEN>
```

3. Run the notebook cells. When the Flask server starts the notebook prints the public ngrok URL; use that URL as the backend.

## Backend — Local (Jupyter)
1. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required packages:

```bash
pip install Flask pyngrok flask_cors langchain chromadb llama_cpp_python huggingface_hub
```

3. Start Jupyter and open the notebook:

```bash
jupyter notebook "hi_Fin_Chatbot (1).ipynb"
```

4. Run the notebook cells. The notebook prints the public ngrok URL when the Flask server starts.

## Flutter client
1. Change to the Flutter project directory:

```bash
cd ai1/ai1
```

2. Fetch packages:

```bash
flutter pub get
```

3. Run the app (replace placeholders):

```bash
flutter run --dart-define=BACKEND_URL=https://your-ngrok-url --dart-define=FINNHUB_API_KEY=your_key
```


