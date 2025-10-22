# Gemini Tool-Using Weather Agent (API)

This is an AI agent built with Python, FastAPI, and the Gemini API (`google-genai` library). It can understand natural language queries and use a "tool" to fetch real-time weather data from the OpenWeatherMap API.

This project demonstrates the "ReAct" (Reasoning and Acting) pattern, where an LLM is used not only to generate text but also to decide which actions (tools) to take.

## Features

-   **Natural Language Understanding:** Uses `gemini-2.5-flash` to parse user intent and parameters.
-   **Tool Use:** Capable of calling a `get_weather` function to fetch live data.
-   **API-based:** Built with FastAPI, exposing an `/agent` endpoint.
-   **Error Handling:** Includes robust error handling for API failures, missing locations, invalid model names, and API version issues.
-   **Modern SDK:** Uses the current `google-genai` library and `v1` API.

## How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
    cd YOUR-REPO-NAME
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Get API Keys:**
    -   Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    -   Get a free API key from [OpenWeatherMap](https://openweathermap.org/).

4.  **Create a `.env` file** and add your keys:
    ```dotenv
    GEMINI_API_KEY=YOUR_GEMINI_KEY
    OPENWEATHERMAP_API_KEY=YOUR_OPENWEATHERMAP_KEY
    ```

5.  **Run the FastAPI server:**
    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

6.  **Test the agent (in a separate terminal):**
    ```bash
    curl -X POST [http://127.0.0.1:8000/agent](http://127.0.0.1:8000/agent) -H "Content-Type: application/json" -d '{"query": "What is the weather in Kharagpur?"}'
    ```
    ```bash
    curl -X POST [http://127.0.0.1:8000/agent](http://127.0.0.1:8000/agent) -H "Content-Type: application/json" -d '{"query": "Tell me a joke"}'
    ```
