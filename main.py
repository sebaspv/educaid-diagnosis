from fastapi import FastAPI
import uvicorn
import openai
import os
from dotenv import dotenv_values
import json
from transformers import pipeline

zeroshot = pipeline(model="facebook/bart-large-mnli")

config = dotenv_values(".env")
openai.api_key = config["API_KEY"]

app = FastAPI()

f = open('data/questions.json')
data = json.load(f)
def patient_description(answer):
    description = ""
    if answer[0] == 1:
        description+="Patient has trouble developing social ties "
    if answer[1] > 5:
        description+="Patient has difficulty identifying emotions of others "
    if answer[2] > 5:
        description+="Patient has difficulty maintaining eye contact "
    if 5 > answer[3]:
        description+="Patient dislikes physical contact "
    if answer[5] > 5:
        description+="Patient dislikes change "
    if (answer[9] > 5) or (answer[7] > 5):
        description+="Patient is sensitive to textures or noises "
    if answer[10] > 5:
        description+="Patient has difficulty studying "
    if answer[12] > 5:
        description+="Patient often forgets things "
    if answer[13] > 5:
        description+="Patient cannot stay still "
    if answer[16] > 5:
        description+="Patient cannot read long texts with ease "
    if answer[17] > 5:
        description+="Patient makes spelling mistakes often "
    if answer[22] > 5:
        description+="Patient often feels in another reality "
    if answer[23] > 5:
        description+="Patient expresses themselves with difficulty "
    return description

@app.get("/explain")
def explain(subject: str, condition: str, interest: str) -> dict:
    """Explain subject based on a list of interests"""
    base_text = f"Explain {subject} to a {condition} patient interested in {interest}"
    completion = openai.Completion.create(
        model="text-davinci-003",
        prompt=base_text,
        temperature=0,
        max_tokens=100
    )
    return completion

@app.post("/diagnosis")
def diagnosis(answers: list) -> str:
    description = patient_description(answers)
    classification = zeroshot(description,
    candidate_labels=["autism", "adhd", "dyslexia", "schizophrenia", "nothing"],
    )
    return str({"diagnosis": classification["sequence"].capitalize(), "condition": classification["labels"][0]})
