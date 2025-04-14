import json
import ollama
import asyncio
from flask import Flask, request, jsonify
import uuid

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
def get_hospital_timings():
    return json.dumps({
        "operating_days": "Monday to Friday",
        "opening_time": "8:00 AM",
        "closing_time": "6:00 PM",
        "closed_days": "Saturday and Sunday"
    })

# Function to return hospital address
def get_hospital_address():
    return json.dumps({
        "name": "MediCare General Hospital",
        "street": "1234 Wellness Avenue",
        "city": "Springfield",
        "state": "IL",
        "zipcode": "62704",
        "country": "USA"
    })

# Function to handle prescription refill requests
def refill_prescription(doctor_name, medication_name, quantity, patient_name):
    doctor_name_lower = doctor_name.lower()
    matched_doctor = next((doc for doc in doctors if doctor_name_lower in doc["name"].lower()), None)

    if not matched_doctor:
        return json.dumps({"error": "Doctor not found. Please provide a valid doctor name."})

    # In a real system, we'd store or process this request
    return json.dumps({
        "status": "success",
        "message": f"Prescription refill request submitted for {patient_name}.",
        "doctor": matched_doctor["name"],
        "medication": medication_name,
        "quantity": quantity
    })

# Function to return doctor details
def get_doctor_details(name):
    query_name = name.lower()
    matches = [doctor for doctor in doctors if query_name in doctor["name"].lower()]

    if len(matches) == 1:
        return json.dumps(matches[0])
    elif len(matches) > 1:
        return json.dumps({"multiple_matches": [doctor["name"] for doctor in matches]})
    return json.dumps({"error": "Doctor not found"})

conversation_history = {}

system_prompt = {
    "role": "system",
    "content": (
        "You are an intelligent IVR assistant for a hospital. Answer politely and professionally. "
        "Use provided functions for hospital timings, address details, or doctor information when needed. "
        "If multiple doctors match the name provided, ask the user to specify the full name."
        "Exactly request the correct parameters for the function call if required, do not make up any parameters. follow the tools properly."
    )
}

client = ollama.AsyncClient()

# Maximum retry attempts for failed function calls
MAX_RETRY_ATTEMPTS = 3  # Configurable number of retries


# Corrected final_check to clean conversation history and retry failed function calls
async def final_check(call_sid):
    available_functions = {
        "get_hospital_timings": get_hospital_timings,
        "get_hospital_address": get_hospital_address,
        "get_doctor_details": get_doctor_details,
        "refill_prescription": refill_prescription,
    }

    # Identify and store failed tool calls before removing them
    failed_tool_calls = [
        msg for msg in conversation_history[call_sid]
        if msg.get("role") == "tool" and "error" in msg.get("content", "").lower()
    ]

    # Remove error tool responses from conversation history
    conversation_history[call_sid] = [
        msg for msg in conversation_history[call_sid]
        if not (msg.get("role") == "tool" and "error" in msg.get("content", "").lower())
    ]

    updated_tool_calls = []
    for message in failed_tool_calls:
        tool_call_id = message.get("tool_call_id", "unknown_id")

        # Corrected: Get original tool response, not assistant
        original_call = next(
            (msg for msg in conversation_history[call_sid]
             if msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id),
            None
        )

        if original_call:
            function_name = original_call.get("function", {}).get("name")
            arguments = original_call.get("function", {}).get("arguments", {})
            function_to_call = available_functions.get(function_name)


            if function_to_call:
                # Attempt multiple retries if necessary
                retry_count = 0
                function_response = None
                success = False

                while retry_count < MAX_RETRY_ATTEMPTS and not success:
                    try:
                        # Retry the function call
                        function_response = function_to_call(**arguments) if arguments else function_to_call()
                        success = True
                        updated_tool_calls.append({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": function_response,
                        })
                    except Exception as e:
                        retry_count += 1
                        success = False
                        if retry_count >= MAX_RETRY_ATTEMPTS:
                            # If all retries fail, append the error
                            updated_tool_calls.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps({"error": f"Retry failed after {MAX_RETRY_ATTEMPTS} attempts: {str(e)}"})
                            })

    # Add retried results to conversation history
    conversation_history[call_sid].extend(updated_tool_calls)


async def generate_response(model: str, call_sid: str):
    available_functions = {
        "get_hospital_timings": get_hospital_timings,
        "get_hospital_address": get_hospital_address,
        "get_doctor_details": get_doctor_details,
    }

    response = await client.chat(
        model=model,
        messages=conversation_history[call_sid],
        tools=[
            {"type": "function", "function": {"name": "get_hospital_timings",
                                              "description": "Provides operating hours of the hospital",
                                              "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "get_hospital_address",
                                              "description": "Provides the full address of the hospital",
                                              "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "get_doctor_details",
                                              "description": "Provides details of a specified doctor",
                                              "parameters": {"type": "object", "properties": {"name": {
                                                  "type": "string", "description": "Doctor's name"}}},
                                              "required": ["name"]}},
            {"type": "function", "function": {
                "name": "refill_prescription",
                "description": "Handles prescription refill requests by verifying doctor and processing the refill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {"type": "string", "description": "Name of the prescribing doctor"},
                        "medication_name": {"type": "string", "description": "Name of the medication"},
                        "quantity": {"type": "string", "description": "Amount of medication requested"},
                        "patient_name": {"type": "string", "description": "Name of the patient"}
                    },
                    "required": ["doctor_name", "medication_name", "quantity", "patient_name"]
                }
            }}
        ],
    )

    conversation_history[call_sid].append({
        "role": response["message"].role,
        "content": response["message"].content
    })

    for tool in response["message"].get("tool_calls", []):
        tool_call_id = str(uuid.uuid4())  # Generate unique ID
        function_name = tool["function"]["name"]
        arguments = tool["function"].get("arguments", {})
        function_to_call = available_functions.get(function_name)

        # Add tool call information to the conversation history
        conversation_history[call_sid].append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "function": {
                "name": function_name,
                "arguments": arguments
            }
        })

        if function_to_call:
            try:
                function_response = function_to_call(**arguments) if arguments else function_to_call()
                conversation_history[call_sid].append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": function_response,
                })
            except Exception as e:
                conversation_history[call_sid].append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps({"error": f"Function execution failed: {str(e)}"})
                })

    # Final check to clean history and retry errors
    await final_check(call_sid)

    # Generate final response only once after corrections
    final_response = await client.chat(model=model, messages=conversation_history[call_sid])
    conversation_history[call_sid].append({"role": "assistant", "content": final_response["message"]["content"]})

    return final_response["message"]["content"], conversation_history[call_sid]


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    call_sid = data.get("call_sid")
    user_input = data.get("user_input", "")

    if call_sid not in conversation_history:
        conversation_history[call_sid] = [system_prompt]

    conversation_history[call_sid].append({"role": "user", "content": user_input})

    response, updated_conversation = loop.run_until_complete(generate_response("llama3.2", call_sid))

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(debug=True)
