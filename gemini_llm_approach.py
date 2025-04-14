import os
import json
import asyncio
import uuid
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import json
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase (do this once)
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")  # Replace with your actual path
    firebase_admin.initialize_app(cred)

db = firestore.client()

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


# Function to handle prescription refill
def request_prescription_refill(patient_name, doctor_name, medicine, dosage):
    try:
        doctor_info = get_doctor_details(doctor_name)
        if "error" in doctor_info or "multiple_matches" in doctor_info:
            return {
                "error": "Doctor not found or multiple matches. Please provide the full doctor name."
            }

        now = datetime.utcnow()
        refill_data = {
            "patient_name": patient_name,
            "medicine_name": medicine,
            "dosage": dosage,
            "doctor_name": doctor_info["name"],
            "status": "order_confirmed",
            "created_at": now.isoformat(),
            "updated_at": now.isoformat()
        }

        # Add to Firestore
        doc_ref = db.collection("prescription_refill_requests").add(refill_data)
        print(f"Added document with ID: {doc_ref[1].id}")
        doc_id = doc_ref[1].id  

        return {
            "confirmation": f"Prescription refill recorded for patient {patient_name} with prescription ID {doc_id}.",
            "prescription_id": doc_id,
            "details": refill_data
        }
    except Exception as e:
        print(f"Error: {str(e)}")



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
            text="""You are an intelligent IVR assistant for a hospital. 
                 Answer politely and professionally. Use provided functions for hospital timings, address details, or doctor information when needed. 
                 If multiple doctors match the name provided, ask the user to specify the full name.
                 If a prescription refill request has been placed successfully and the request data is given as input, tell the user that it has been successfully places and give the user the prescription_id of the request.
                 Finally answer anything again if the user asks a question again."""
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
    },
    {
        "name": "request_prescription_refill",
        "description": "Handles a prescription refill request after verifying the doctor.",
        "parameters": {
            "type": "object",
            "properties": {
                "patient_name": {
                    "type": "string",
                    "description": "Full name of the patient"
                },
                "doctor_name": {
                    "type": "string",
                    "description": "Full name of the prescribing doctor"
                },
                "medicine": {
                    "type": "string",
                    "description": "Name of the medicine requested"
                },
                "dosage": {
                    "type": "string",
                    "description": "Dosage amount"
                }
            },
            "required": ["patient_name", "doctor_name", "medicine", "dosage"]
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
    "request_prescription_refill": request_prescription_refill,
}

# Retry logic for failed function calls
MAX_RETRY_ATTEMPTS = 2


async def final_check(call_sid):
    """ Retry failed function calls if necessary """
    failed_tool_calls = [
        msg for msg in conversation_history[call_sid]
        if msg.role == "user" and "error" in msg.parts[0].text.lower()
    ]

    # Remove error responses from conversation history
    conversation_history[call_sid] = [
        msg for msg in conversation_history[call_sid]
        if not (msg.role == "user" and "error" in msg.parts[0].text.lower())
    ]

    updated_tool_calls = []
    for message in failed_tool_calls:
        tool_call_id = json.loads(message.parts[0].text).get("tool_call_id")
        original_call = next(
            (msg for msg in conversation_history[call_sid]
             if msg.role == "model" and tool_call_id in str(msg.parts)),
            None
        )

        if original_call:
            function_name = json.loads(original_call.parts[0].text).get("function").get("name")
            arguments = json.loads(original_call.parts[0].text).get("function").get("arguments", {})
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
                            types.Content(role="user", parts=[types.Part(text=json.dumps(function_response))])
                        )
                    except Exception as e:
                        retry_count += 1
                        success = False
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
    """ Generate a response using Gemini with function calling support """

    # Use conversation history directly since it's in types.Content format
    user_messages = conversation_history[call_sid]

    # Debug: Print conversation history

    # Generate content using Gemini
    response = client.models.generate_content(
        model=model,
        contents=user_messages,
        config=config
    )

    # Validate if response is valid
    if not response.candidates or not response.candidates[0].content.parts:
        raise ValueError("Empty or invalid response from Gemini")

    # ✅ Loop through all parts returned in the response
    for part in response.candidates[0].content.parts:
        if part.function_call:
            function_call_data = part.function_call
            function_name = function_call_data.name
            arguments = function_call_data.args
            tool_call_id = str(uuid.uuid4())


            # Add tool call information to the conversation history
            conversation_history[call_sid].append(
                types.Content(
                    role="model",  # Use 'model' to log tool calls
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
                            role="user",
                            parts=[types.Part(text=json.dumps(function_response))]  # Append function response to conversation history
                        )
                    )
                except Exception as e:
                    conversation_history[call_sid].append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=json.dumps({"error": f"Function execution failed: {str(e)}", "tool_call_id": tool_call_id}))]
                        )
                    )
            else:
                # If function not found, log an error
                conversation_history[call_sid].append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=json.dumps({"error": f"Unknown function '{function_name}'","tool_call_id": tool_call_id}))]
                    )
                )
        else:
            # If no function_call, treat as regular text and append to conversation history
            conversation_history[call_sid].append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=part.text)]
                )
            )

            return part.text, conversation_history[call_sid]

    # ✅ Run final check to retry any errors or failed function calls
    await final_check(call_sid)
                   

    # ✅ Generate final response after processing function calls
    final_response = client.models.generate_content(
        model=model,
        contents=conversation_history[call_sid],
        config=config
    )

   
    # ✅ Loop again for the final response if needed
    final_output = []
    for part in final_response.candidates[0].content.parts:
        if part.function_call:
            final_function_call_data = part.function_call
            function_name = final_function_call_data.name
            arguments = final_function_call_data.args
            tool_call_id = str(uuid.uuid4())

            # Check if the function is available
            function_to_call = available_functions.get(function_name)

            if function_to_call:
                try:
                    # Call the function and get the result
                    function_response = function_to_call(**arguments) if arguments else function_to_call()
                    conversation_history[call_sid].append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=json.dumps(function_response))]  # Append function response to conversation history
                        )
                    )
                    final_output.append(json.dumps(function_response))
                except Exception as e:
                    conversation_history[call_sid].append(
                        types.Content(
                            role="user",
                            parts=[types.Part(text=json.dumps({"error": f"Function execution failed: {str(e)}","tool_call_id": tool_call_id}))]
                        )
                    )
                    final_output.append(json.dumps({"error": f"Function execution failed: {str(e)}", "tool_call_id": tool_call_id}))
            else:
                conversation_history[call_sid].append(
                    types.Content(
                        role="user",
                        parts=[types.Part(text=json.dumps({"error": f"Unknown function '{function_name}'", "tool_call_id": tool_call_id}))]
                    )
                )
                final_output.append(json.dumps({"error": f"Unknown function '{function_name}'"}))
        else:
            # Treat as normal text and append to final output
            final_output.append(part.text)

    final_reply = "\n".join(final_output)

    conversation_history[call_sid].append(
        types.Content(
            role="model",
            parts=[types.Part(text=final_reply)]
        )
    )

    # ✅ Return combined results and updated conversation history
    return final_reply, conversation_history[call_sid]



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
