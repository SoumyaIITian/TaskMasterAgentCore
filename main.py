import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.genai as genai
import google.genai.types as types
from dotenv import load_dotenv
import json 

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")

if not GEMINI_API_KEY or not WEATHER_API_KEY:
    raise ValueError("API keys for Gemini and OpenWeatherMap must be set in .env file")


try:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(api_version='v1')
    )
    
    generation_config = types.GenerateContentConfig(
        temperature=0.6,
        top_p=1,
        top_k=1,
        max_output_tokens=2048
    )
    
    # Use a modern, stable model.
    model_name = "models/gemini-2.5-flash" 
    
    print(f"Gemini client configured successfully for model: {model_name}")
except Exception as e:
    print(f"Error configuring Gemini model: {e}")
    raise

app = FastAPI()

class UserRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    response: str
    debug_info: dict = None 

# --- Tool Functions ---
def get_weather(location: str):
    """
    Fetches current weather data for a specified location using OpenWeatherMap API.
    Returns a formatted string or an error message.
    """
    print(f"--- Calling Weather Tool for location: '{location}' ---")
    if not location or location.strip() == "":
         print("--- Weather Tool Error: Location cannot be empty ---")
         return "Error: Please specify a location for the weather."
         
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + WEATHER_API_KEY + "&q=" + requests.utils.quote(location) + "&units=metric"
    
    try:
        response = requests.get(complete_url, timeout=10) 
        if response.status_code >= 500:
            response.raise_for_status() # Raises HTTPError for bad responses (4XX, 5XX)
        data = response.json()

        if data.get("cod") == 200: 
            main = data.get("main", {})
            weather_list = data.get("weather", [])
            
            temperature = main.get("temp", "N/A")
            feels_like = main.get("feels_like", "N/A")
            humidity = main.get("humidity", "N/A")
            description = weather_list[0].get("description", "N/A") if weather_list else "N/A"
            city_name = data.get("name", location) # Use name from response for confirmation
            
            result = (
                f"The weather in {city_name} is currently {description} "
                f"with a temperature of {temperature}°C (feels like {feels_like}°C) "
                f"and humidity of {humidity}%."
            )
            print(f"--- Weather Tool Result: {result} ---")
            return result
        elif data.get("cod") == "404":
             print(f"--- Weather Tool Error: City not found ('{location}') ---")
             return f"Error: Sorry, I couldn't find weather data for the city '{location}'. Please check the spelling."
        else:
            # Handle other potential API errors reported by OpenWeatherMap
            error_message = data.get("message", "Unknown API error")
            print(f"--- Weather Tool Error: API returned code {data.get('cod')} - {error_message} ---")
            return f"Error: Could not retrieve weather data. API reported: {error_message}"

    except requests.exceptions.Timeout:
        print("--- Weather Tool Error: API request timed out ---")
        return "Error: The weather service took too long to respond. Please try again later."
    except requests.exceptions.RequestException as e:
        print(f"--- Weather Tool Error: API request failed - {e} ---")
        return f"Error: Could not connect to the weather service. Please check your connection or try again later."
    except KeyError as e:
        print(f"--- Weather Tool Error: Unexpected API response format - Missing key {e} ---")
        return "Error: Received an unexpected response format from the weather service."
    except Exception as e: # Catch any other unexpected errors
        print(f"--- Weather Tool Error: An unknown error occurred - {e} ---")
        # In production, log the full traceback here
        return "Error: An unexpected error occurred while fetching weather data."

# --- Agent Orchestrator Logic ---
# Define available tools
TOOLS = {
    "get_weather": get_weather,
}

async def run_agent(user_query: str):
    """
    Orchestrates the agent's response using Gemini API and tools.
    """
    print(f"\n--- Starting Agent Run for Query: '{user_query}' ---")
    
    tool_to_use = None
    parameters = {}
    tool_result = None
    
    # 1. Intent Parsing & Planning (using Gemini)
    #    Prompt Gemini to identify the goal, required tool, and parameters.
    #    We'll ask it to respond in a structured format (JSON).
    intent_prompt = f"""
    Analyze the user's query and determine the primary intent. 
    Based on the intent, decide if one of the available tools should be used.
    Available tools: {list(TOOLS.keys())}
    
    User Query: "{user_query}"

    Respond in JSON format with the following keys:
    - "tool_name": The name of the tool to use (from the available list) or "none" if no tool is needed or the intent is unclear.
    - "parameters": An object containing the necessary parameters for the tool (e.g., {{"location": "City Name"}} for get_weather). If no tool is needed, this should be an empty object {{}}.

    Example 1:
    User Query: "What's the weather like in London?"
    Response: {{"tool_name": "get_weather", "parameters": {{"location": "London"}}}}

    Example 2:
    User Query: "Tell me a joke."
    Response: {{"tool_name": "none", "parameters": {{}}}}
    
    Example 3:
    User Query: "What is the capital of France?"
    Response: {{"tool_name": "none", "parameters": {{}}}} # No specific tool for general knowledge

    Example 4: 
    User Query: "How's the weather?"
    Response: {{"tool_name": "get_weather", "parameters": {{}}}} # Tool identified, but parameter missing
    """
    
    try:
        print("--- Calling Gemini for Intent Parsing ---")
        # Make the Gemini call synchronous within the async function for simplicity here
        # For production, consider fully async Gemini calls if performance is critical
        intent_response = client.models.generate_content(
            model=model_name,
            contents=intent_prompt,
            config=generation_config  # <-- THE FIX
        )
        # Clean potential markdown/formatting issues
        cleaned_response_text = intent_response.text.strip().replace('```json', '').replace('```', '').strip()
        print(f"--- Gemini Intent Raw Response: {cleaned_response_text} ---")
        
        parsed_intent = json.loads(cleaned_response_text)
        tool_to_use = parsed_intent.get("tool_name")
        parameters = parsed_intent.get("parameters", {})
        print(f"--- Parsed Intent: Tool='{tool_to_use}', Params={parameters} ---")

    except json.JSONDecodeError:
        print("--- Gemini Intent Parsing Error: Failed to parse JSON response ---")
        # Fallback: Let the LLM handle it directly later
        tool_to_use = "none" 
    except Exception as e:
        print(f"--- Gemini Intent Parsing Error: {e} ---")
        # Fallback: Let the LLM handle it directly later
        tool_to_use = "none"

    # 2. Tool Execution (if a valid tool and parameters were identified)
    if tool_to_use and tool_to_use in TOOLS:
        tool_function = TOOLS[tool_to_use]
        required_params = tool_function.__code__.co_varnames[:tool_function.__code__.co_argcount]
        
        # Check if all required parameters are present
        if all(param in parameters for param in required_params):
             try:
                # Call the actual tool function with the extracted parameters
                # Making this synchronous for now
                tool_result = tool_function(**parameters) 
             except Exception as e:
                 print(f"--- Tool Execution Error ({tool_to_use}): {e} ---")
                 tool_result = f"Error: Failed to execute the tool '{tool_to_use}'."
        else:
             print(f"--- Tool Execution Error: Missing parameters for {tool_to_use}. Required: {required_params}, Got: {parameters} ---")
             # Specific error message if parameters are missing (LLM will use this)
             tool_result = f"Error: I need more information to use the '{tool_to_use}' tool. Please provide: {', '.join(required_params)}."
             # Set tool_to_use back to none so the final prompt knows parameters were missing
             tool_to_use = "none_missing_params" 
             
    elif tool_to_use and tool_to_use != "none":
        print(f"--- Tool Execution Error: Tool '{tool_to_use}' is not available. ---")
        tool_result = f"Error: I cannot perform that action as the tool '{tool_to_use}' is not available."
        tool_to_use = "none" # Treat as no tool executed

    # 3. Response Generation (using Gemini, potentially with tool result)
    final_response_prompt = ""
    if tool_result:
        # Provide the tool result back to the LLM
        final_response_prompt = f"""
        User Query: "{user_query}"
        You decided to use the tool: '{parsed_intent.get("tool_name", "N/A")}' with parameters: {parameters}
        The result from the tool is: "{tool_result}"
        
        Based ONLY on the user query and the tool result, generate a concise and helpful natural language response for the user. 
        - If the tool executed successfully, incorporate its result smoothly into your answer.
        - If the tool returned an error (e.g., city not found, missing parameters, API failure), explain the error clearly and politely to the user. Do not make up information if there was an error.
        """
    elif tool_to_use == "none_missing_params":
         # Handle the case where the LLM knew the tool but parameters were missing
         final_response_prompt = f"""
         User Query: "{user_query}"
         You identified the intent requires the '{parsed_intent.get("tool_name", "N/A")}' tool, but the necessary parameters ({', '.join(required_params)}) were missing in the query.
         Politely ask the user to provide the missing information. For example, if they asked for weather without a city, ask them for the city name.
         """
    else:
        # If no tool was needed, or intent parsing failed completely
        final_response_prompt = f"""
        User Query: "{user_query}"
        No specific tool was needed or could be confidently identified to answer this query.
        Respond directly and conversationally to the user based on their query. If it's general knowledge you might know, answer it. If not, state that you cannot perform that specific task or ask for clarification.
        """

    try:
        print("--- Calling Gemini for Final Response Generation ---")
        # Again, synchronous call for simplicity
        final_response_part = client.models.generate_content(
            model=model_name,
            contents=final_response_prompt,
            config=generation_config  # <-- THE FIX
        )
        final_answer = final_response_part.text.strip()
        print(f"--- Gemini Generated Final Answer: {final_answer} ---")
    except Exception as e:
        print(f"--- Gemini Final Response Generation Error: {e} ---")
        final_answer = "Sorry, I encountered an error while trying to generate a response."

    return {
        "response": final_answer,
        "debug_info": { 
            "parsed_intent": parsed_intent if 'parsed_intent' in locals() else "Parsing Failed",
            "tool_called": parsed_intent.get("tool_name", "N/A") if tool_result else "none",
            "tool_parameters": parameters,
            "tool_result": tool_result if tool_result else "N/A"
        }
    }

# --- API Endpoint ---
@app.post("/agent", response_model=AgentResponse)
async def handle_agent_request(request: UserRequest):
    """Receives user query and returns the agent's response."""
    if not request.query or request.query.strip() == "":
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = await run_agent(request.query)
        return AgentResponse(response=result["response"], debug_info=result.get("debug_info"))
    except Exception as e:
        print(f"--- API Endpoint Error: {e} ---")
        # Log the full error traceback here in a real application
        raise HTTPException(status_code=500, detail=f"Internal Server Error: An unexpected error occurred.")

# --- Main Execution (for local testing/running) ---
if __name__ == "__main__":
    import uvicorn
    print("Starting TaskMaster Agent Core...")
    print("API keys loaded.")
    print("Run using: uvicorn main:app --reload --host 0.0.0.0 --port 8000")
