import os
import json
import asyncio
import uuid
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Doctors list
doctors = [
    {"name": "John Doe", "department": "Pulmonology", "specialization": "Surgery",
     "timings": "Monday to Friday, 3:00 pm to 5:00 pm"},
    {"name": "Jane Smith", "department": "Cardiology", "specialization": "Heart Surgery",
     "timings": "Monday, Wednesday, Friday, 10:00 am to 12:00 pm"},
    {"name": "Emily Johnson", "department": "Neurology", "specialization": "Brain Surgery",
     "timings": "Tuesday and Thursday, 1:00 pm to 3:00 pm"},
    {"name": "Michael Brown", "department": "Orthopedics", "specialization": "Knee Replacement",
     "timings": "Monday to Friday, 9:00 am to 11:00 am"},
    {"name": "Sarah Lee", "department": "Dermatology", "specialization": "Skin Treatment",
     "timings": "Monday to Friday, 2:00 pm to 4:00 pm"},
    {"name": "William Clark", "department": "Ophthalmology", "specialization": "Cataract Surgery",
     "timings": "Monday to Friday, 10:00 am to 12:00 pm"},
    {"name": "John Smith", "department": "Pediatrics", "specialization": "Child Care",
     "timings": "Monday to Friday, 11:00 am to 1:00 pm"}
]


# Function to return hospital timings
def get_hospital_timings(dummy_param=None):
    return {
        "operating_days": "Monday to Friday",
        "opening_time": "8:00 AM",
        "closing_time": "6:00 PM",
        "closed_days": "Saturday and Sunday"
    }


# Function to return hospital address
def get_hospital_address(dummy_param=None):
    return {
        "name": "MediCare General Hospital",
        "street": "1234 Wellness Avenue",
        "city": "Springfield",
        "state": "IL",
        "zipcode": "62704",
        "country": "USA"
    }


# Function to return doctor details
def get_doctor_details(name):
    query_name = name.lower()
    matches = [doctor for doctor in doctors if query_name in doctor["name"].lower()]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return {"multiple_matches": [doctor["name"] for doctor in matches]}
    return {"error": "Doctor not found"}


# Conversation history
conversation_history = {}

# Corrected system prompt in types.Content format
system_prompt = types.Content(
    role="user",  # Using 'user' for system prompt as 'system' is not supported by Gemini
    parts=[
        types.Part(
            text="You are an intelligent IVR assistant for a hospital. "
                 "Answer politely and professionally. Use provided functions "
                 "for hospital timings, address details, or doctor information "
                 "when needed. If multiple doctors match the name provided, "
                 "ask the user to specify the full name."
        )
    ]
)

# Gemini API configuration
genai_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=genai_api_key)

# Define function declarations for Gemini
functions = [
    {
        "name": "get_hospital_timings",
        "description": "Provides operating hours of the hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "dummy_param": {
                    "type": "string",
                    "description": "Dummy parameter, not used."
                }
            },
            "required": []
        },
    },
    {
        "name": "get_hospital_address",
        "description": "Provides the full address of the hospital.",
        "parameters": {
            "type": "object",
            "properties": {
                "dummy_param": {
                    "type": "string",
                    "description": "Dummy parameter, not used."
                }
            },
            "required": []
        },
    },
    {
        "name": "get_doctor_details",
        "description": "Provides details of a specified doctor.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Doctor's name to retrieve details."
                }
            },
            "required": ["name"]
        }
    }
]

tools = types.Tool(function_declarations=functions)
config = types.GenerateContentConfig(tools=[tools])

# Available functions mapping
available_functions = {
    "get_hospital_timings": get_hospital_timings,
    "get_hospital_address": get_hospital_address,
    "get_doctor_details": get_doctor_details,
}

# Retry logic for failed function calls
MAX_RETRY_ATTEMPTS = 3


async def final_check(call_sid):
    """ Retry failed function calls if necessary """
    failed_tool_calls = [
        msg for msg in conversation_history[call_sid]
        if msg.role == "model" and "error" in msg.parts[0].text.lower()
    ]

    # Remove error responses from conversation history
    conversation_history[call_sid] = [
        msg for msg in conversation_history[call_sid]
        if not (msg.role == "model" and "error" in msg.parts[0].text.lower())
    ]

    updated_tool_calls = []
    for message in failed_tool_calls:
        tool_call_id = str(uuid.uuid4())
        original_call = next(
            (msg for msg in conversation_history[call_sid]
             if msg.role == "user" and tool_call_id in str(msg.parts)),
            None
        )

        if original_call:
            function_name = original_call.parts[0].text.split("(")[0]
            arguments = json.loads(original_call.parts[0].text.split("(")[1].split(")")[0])
            function_to_call = available_functions.get(function_name)

            if function_to_call:
                retry_count = 0
                function_response = None
                success = False

                while retry_count < MAX_RETRY_ATTEMPTS and not success:
                    try:
                        function_response = function_to_call(**arguments) if arguments else function_to_call()
                        success = True
                        updated_tool_calls.append(
                            types.Content(role="model", parts=[types.Part(text=json.dumps(function_response))])
                        )
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= MAX_RETRY_ATTEMPTS:
                            updated_tool_calls.append(
                                types.Content(role="model", parts=[
                                    types.Part(text=json.dumps(
                                        {"error": f"Retry failed after {MAX_RETRY_ATTEMPTS} attempts: {str(e)}"}
                                    ))
                                ])
                            )

    # Add retried results to conversation history
    conversation_history[call_sid].extend(updated_tool_calls)


async def generate_response(model: str, call_sid: str):
    """ Generate a response using Gemini """
    # Use conversation history directly since it's in types.Content format
    user_messages = conversation_history[call_sid]

    # Debug: Print conversation history
    print("Conversation history before API call:", user_messages)

    # Generate content using Gemini
    response = client.models.generate_content(
        model=model,
        contents=user_messages,
        config=config
    )

    # Validate if response is valid
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Empty or invalid response from Gemini")

    # Check if the response contains a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call_data = response.candidates[0].content.parts[0].function_call
        function_name = function_call_data.name
        arguments = function_call_data.args
        tool_call_id = str(uuid.uuid4()) 

        # ✅ Add tool call information to the conversation history
        conversation_history[call_sid].append(
            types.Content(
                role="model",  # Use 'model' for tool call logging
                parts=[types.Part(text=json.dumps({
                    "tool_call_id": tool_call_id,
                    "function": {
                        "name": function_name,
                        "arguments": arguments
                    }
                }))]
            )
        )

        # Check if the function is available
        function_to_call = available_functions.get(function_name)

        if function_to_call:
            try:
                # Call the function and get the result
                function_response = function_to_call(**arguments) if arguments else function_to_call()

                # Add the function result to conversation history
                conversation_history[call_sid].append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps(function_response))]
                    )
                )
            except Exception as e:
                conversation_history[call_sid].append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps({"error": f"Function execution failed: {str(e)}"}))]
                    )
                )
        else:
            # If function not found, log an error
            conversation_history[call_sid].append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=json.dumps({"error": f"Unknown function '{function_name}'"}))]
                )
            )
    else:
        # If no function_call, treat as regular text
        conversation_history[call_sid].append(
            types.Content(
                role="model",
                parts=[types.Part(text=response.candidates[0].content.parts[0].text)]
            )
        )

    # Run final check to retry any errors
    await final_check(call_sid)

    # Generate final response after corrections
    final_response = client.models.generate_content(
        model=model,
        contents=conversation_history[call_sid],
        config=config
    )

    # Handle final response correctly
    if final_response.candidates[0].content.parts[0].function_call:
        final_function_call_data = final_response.candidates[0].content.parts[0].function_call
        function_name = final_function_call_data.name
        arguments = final_function_call_data.args

        # Check if the function is available
        function_to_call = available_functions.get(function_name)

        if function_to_call:
            try:
                # Call the function and get the result
                function_response = function_to_call(**arguments) if arguments else function_to_call()

                # Add the function result to conversation history
                conversation_history[call_sid].append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps(function_response))]
                    )
                )
                return json.dumps(function_response), conversation_history[call_sid]
            except Exception as e:
                conversation_history[call_sid].append(
                    types.Content(
                        role="model",
                        parts=[types.Part(text=json.dumps({"error": f"Function execution failed: {str(e)}"}))]
                    )
                )
                return json.dumps({"error": f"Function execution failed: {str(e)}"}), conversation_history[call_sid]
        else:
            # If function not found, log an error
            conversation_history[call_sid].append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=json.dumps({"error": f"Unknown function '{function_name}'"}))]
                )
            )
            return json.dumps({"error": f"Unknown function '{function_name}'"}), conversation_history[call_sid]
    else:
        conversation_history[call_sid].append(
            types.Content(
                role="model",
                parts=[types.Part(text=final_response.candidates[0].content.parts[0].text)]
            )
        )
        return final_response.candidates[0].content.parts[0].text, conversation_history[call_sid]


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@app.route('/chat', methods=['POST'])
def chat():
    """ Flask endpoint to handle chat requests """
    data = request.json
    call_sid = data.get("call_sid")
    user_input = data.get("user_input", "")

    if call_sid not in conversation_history:
        conversation_history[call_sid] = [
            system_prompt  # Correctly formatted system prompt
        ]

    # Append user input to conversation history
    conversation_history[call_sid].append(
        types.Content(
            role="user",
            parts=[types.Part(text=user_input)]
        )
    )

    # Generate and return response
    response, updated_conversation = loop.run_until_complete(
        generate_response("gemini-2.0-flash", call_sid)
    )

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
