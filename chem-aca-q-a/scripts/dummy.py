from openai import OpenAI

from api_keys.api_keys import key_openai
from .pdf_utils import download_pdf

client = OpenAI(api_key=key_openai)

print("successful import")