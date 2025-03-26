import json
import ollama
import asyncio
from flask import Flask, request, jsonify

app = Flask(__name__)

# Doctors list
doctors = [
    {"name": "John Doe", "department": "Pulmonology", "specialization": "Surgery", "timings": "Monday to Friday, 3:00 pm to 5:00 pm"},
    {"name": "Jane Smith", "department": "Cardiology", "specialization": "Heart Surgery", "timings": "Monday, Wednesday, Friday, 10:00 am to 12:00 pm"},
    {"name": "Emily Johnson", "department": "Neurology", "specialization": "Brain Surgery", "timings": "Tuesday and Thursday, 1:00 pm to 3:00 pm"},
    {"name": "Michael Brown", "department": "Orthopedics", "specialization": "Knee Replacement", "timings": "Monday to Friday, 9:00 am to 11:00 am"},
    {"name": "Sarah Lee", "department": "Dermatology", "specialization": "Skin Treatment", "timings": "Monday to Friday, 2:00 pm to 4:00 pm"},
    {"name": "William Clark", "department": "Ophthalmology", "specialization": "Cataract Surgery", "timings": "Monday to Friday, 10:00 am to 12:00 pm"},
    {"name": "John Smith", "department": "Pediatrics", "specialization": "Child Care", "timings": "Monday to Friday, 11:00 am to 1:00 pm"}
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
    )
}

client = ollama.AsyncClient()

async def generate_response(model: str, call_sid: str):
    response = await client.chat(
        model=model,
        messages=conversation_history[call_sid],
        tools=[
            {"type": "function", "function": {"name": "get_hospital_timings", "description": "Provides operating hours of the hospital", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "get_hospital_address", "description": "Provides the full address of the hospital", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "get_doctor_details", "description": "Provides details of a specified doctor", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "Doctor's name"}}, "required": ["name"]}}}
        ],
    )

    conversation_history[call_sid].append({
        "role": response["message"].role,
        "content": response["message"].content
    })

    available_functions = {
        "get_hospital_timings": get_hospital_timings,
        "get_hospital_address": get_hospital_address,
        "get_doctor_details": get_doctor_details,
    }

    for tool in response["message"].get("tool_calls", []):
        function_name = tool["function"]["name"]
        arguments = tool["function"].get("arguments", {})
        function_to_call = available_functions.get(function_name)
        function_response = function_to_call(**arguments) if arguments else function_to_call()

        conversation_history[call_sid].append({
            "role": "tool",
            "tool_call_id": tool.get("id", "unknown_id"),
            "content": function_response,
        })

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
