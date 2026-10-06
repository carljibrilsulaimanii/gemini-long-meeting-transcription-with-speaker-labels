# list_models.py
# Author: Jibril Sulaiman
# What:   Prints every Gemini model name your API key can call.
# Why:    Model names get retired. When transcribe.py fails with 404 NOT_FOUND
#         ("This model ... is no longer available to new users"), run this to see
#         what your key can use, then set the model in transcribe.py.
# How:    Set GEMINI_API_KEY in your environment, then:  python list_models.py
# Needs:  Python 3.9+, pip install google-genai.

import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

for m in client.models.list():
    print(m.name)
