import json
import traceback
import requests
import fitz
import os
import csv
import re
import subprocess
import openai

from typing import Optional

from api_keys import key_openai

client = openai.OpenAI(
  api_key=key_openai,
)

def download_pdf(url: str, output_folder: str, csv_path: str, folder: str, journal_name: str, article_link: str) -> bool:
    """Download a PDF from a given URL and save it to the specified folder.
    
    Args:
        url (str): The URL of the PDF to download.
        folder (str): The folder to save the downloaded PDF.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(folder, exist_ok=True)
        filename = url.split('/')[-1]
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        print("*"*50)
        print(f'Downloaded {filename}')
        print("*"*50)

        convert_pdf_to_text(filepath, output_folder, csv_path)
        return True
    else:
        print(f'Error {response.status_code} while downloading {url}')
        return False

def convert_pdf_to_text(pdf_path: str, output_folder:str, csv_path: str) -> Optional[str]:
    """Convert a PDF file to text and save it.
    
    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        Optional[str]: The path to the converted text file, or None if conversion fails.
    """
    full_text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_number in range(doc.page_count):
            page = doc[page_number]
            text = page.get_text("text")
            text = re.sub(r'\s+', ' ', text).strip()
            full_text += text + "\n"

        abstract_patterns = [r'\bAbstract\b', r'\bABSTRACT\b', r'\bSummary\b', r'\bExecutive Summary\b', r'\bIntroduction\b']
        references_patterns = [r'\bReferences\b', r'\bREFERENCES\b', r'\bBibliography\b', r'\bWorks Cited\b', r'\bLiterature Cited\b', r'\bReference List\b', r'\bCitations\b']
        abstract_start, references_start = None, None
        for pattern in abstract_patterns:
            abstract_start = re.search(pattern, full_text, re.IGNORECASE)
            if abstract_start:
                break
        for pattern in references_patterns:
            references_start = re.search(pattern, full_text, re.IGNORECASE)
            if references_start:
                break
        if abstract_start and references_start:
            full_text = full_text[abstract_start.start():references_start.start()]
        else:
            print("Abstract or References section not found.")

        title = process_pdf(full_text, pdf_path, csv_path)
        
        new_name = title + ".txt"
        output_path = os.path.join(output_folder, new_name)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(full_text)
    except subprocess.CalledProcessError as e:
        print("*" * 50)
        print(f"Error occurred while converting pdf to txt: {e}")
        print(traceback.format_exc())
        print("*" * 50)
        return None
    return output_path

def process_pdf(text: str, filename: str, csv_path:str) -> str:
    """
    Uses GPT Chat to attempt to extract structured information from a given text,
    explicitly requesting a JSON-formatted response. If extraction fails, it returns a default response.
    
    Args:
        text (str): The text to analyze and extract information from.
        
    Returns:
        str: The title of the paper if extracted successfully, otherwise an error message.
    """
    last_part = filename.split('/')[-1]
    doi = last_part.split('.')[0]
    
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['DOI'] == doi:
                print(f"DOI {doi} already exists in CSV.")
                return doi  # Or an appropriate message

    initial_prompt = f"""Given the following text, extract structured information in JSON format including the title, abstract, authors, keywords, institute of origin, DOI, and funding:
        
        Text: "{text}"
        
        Example Output:
        {{
            "Title": "Example Title",
            "Abstract": "Example abstract text",
            "Journal": "Example journal",
            "Relevant fields": ["field1", "field2", etc], 
            "Authors": ["Author One", "Author Two"],
            "Keywords": ["keyword1", "keyword2"],
            "Institute of Origin": "Example Institute",
            "DOI": "https://doi.org/example",
            "Funding": "Example funding source",
            "Methods": "Detailed information about the experimental setup with specific information about materials, techniques, formulas, numbers, metrics, etc. Write as much as you find.",
            "Results": "Detailed information about the experimental outcomes with specific information about materials, techniques, formulas, numbers, metrics, etc. Write as much as you find.",
            "Experiment details": "Detailed information about the experiment with specific information about materials, techniques, formulas, etc. Write as much as you find."
        }}

        Only output the json, so brackets and wtv is inside. Nothing else.
        """

    messages = [
        {"role": "system", "content": "You are a helpful assistant, skilled in extracting structured information from research papers and outputting it in JSON format."},
        {"role": "user", "content": initial_prompt}
    ]

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )

        last_response = completion.choices[0].message.content
        print("="*10)
        print("Model")
        print(completion.model)
        print("="*10)
        print("Token count")
        print(completion.usage.completion_tokens + int(completion.usage.prompt_tokens))
        print("="*10)
        print(last_response)

        try:
            parsed_response = json.loads(last_response)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return "Error in JSON decoding"
        
        print(parsed_response)
        # Save structured information into a JSON file
        json_output_path = "../data/all_output/" + doi + ".json"
        
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(parsed_response, json_file, indent=4)


        authors = ', '.join(parsed_response['Authors'])
        keywords = ', '.join(parsed_response['Keywords'])
        relevant_fields = ', '.join(parsed_response.get('Relevant fields', []))  # Handles optional fields gracefully

        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'DOI', 'Title', 'Abstract', 'Journal', 'Relevant fields',
                'Authors', 'Keywords', 'Institute of Origin', 'Funding',
                'Methods', 'Results', 'Experiment details'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            data_to_write = {
                'DOI': parsed_response['DOI'],
                'Title': parsed_response['Title'],
                'Abstract': parsed_response['Abstract'],
                'Journal': parsed_response.get('Journal', 'N/A'),  # Using .get() for optional fields
                'Relevant fields': relevant_fields,
                'Authors': authors,
                'Keywords': keywords,
                'Institute of Origin': parsed_response['Institute of Origin'],
                'Funding': parsed_response['Funding'],
                'Methods': parsed_response.get('Methods', 'N/A'),
                'Results': parsed_response.get('Results', 'N/A'),
                'Experiment details': parsed_response.get('Experiment details', 'N/A')
            }

            writer.writerow(data_to_write)

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return "Error in JSON decoding"

    print("Done with ", doi)
    print("*"*50)
    return doi
