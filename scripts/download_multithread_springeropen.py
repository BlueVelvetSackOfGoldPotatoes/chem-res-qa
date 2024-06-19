import json
import sys
import traceback
import openai
import requests
import fitz
import os
import csv
import re
import subprocess
from typing import List, Tuple, Optional
from api_keys.gpt_api import key_openai
from total_cost import TOTAL_COST
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

vpn_locations: List[str] = ['Berlin', 'Germany', 'Munich', 'France', 'Frankfurt', 'Italy', 'Paris', 'Spain', 'Marseille', 'Lyon', 'Rome', 'Milan', 'Naples', 'Amsterdam', 'Rotterdam', 'The Hague', 'Stockholm', 'Gothenburg', 'Malmo', 'Oslo', 'Bergen', 'Trondheim', 'Copenhagen', 'Aarhus', 'Odense', 'Helsinki', 'Espoo', 'Tampere', 'Warsaw', 'Krakow', 'Lodz', 'Vienna', 'Graz', 'Linz', 'Brussels', 'Antwerp', 'Ghent', 'Zurich', 'Geneva', 'Basel', 'Lisbon', 'Porto', 'Braga', 'Athens', 'Thessaloniki', 'Patras', 'Budapest', 'Debrecen', 'Szeged', 'Prague', 'Brno', 'Ostrava', 'Dublin', 'Cork', 'Limerick', 'London', 'Manchester', 'Birmingham']

OUTPUT_FOLDER: str = "./all_output"
CSV_PATH: str = "./downloaded_articles.csv"

client = openai.OpenAI(
  api_key=key_openai,
)

def convert_pdf_to_text(pdf_path: str) -> Optional[str]:
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

        title = process_pdf(full_text, pdf_path)
        
        new_name = title + ".txt"
        output_path = os.path.join(OUTPUT_FOLDER, new_name)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(full_text)
    except subprocess.CalledProcessError as e:
        print("*" * 50)
        print(f"Error occurred while converting pdf to txt: {e}")
        print(traceback.format_exc())
        print("*" * 50)
        return None
    return output_path

def process_pdf(text: str, filename: str) -> str:
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
    
    with open(CSV_PATH, 'r', newline='', encoding='utf-8') as csvfile:
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

        # Extracting the message content properly
        last_response = completion.choices[0].message.content
        print("="*10)
        print("Model")
        print(completion.model)
        print("="*10)
        print("Token count")
        print(completion.usage.completion_tokens + int(completion.usage.prompt_tokens))
        print("="*10)
        total_cost_paper = (int(completion.usage.completion_tokens) * 0.0015 / 1000) + (int(completion.usage.prompt_tokens) * 0.0005 / 1000) # https://openai.com/pricing
        print(f"Total cost for paper: {total_cost_paper}")
        print(f"Incremented cost: {TOTAL_COST + total_cost_paper}")
        print("="*10)

        with open("total_cost.py", "w") as f:
            f.write(f"TOTAL_COST = {repr(TOTAL_COST + total_cost_paper)}\n")

        print(last_response)

        try:
            parsed_response = json.loads(last_response)
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
            return "Error in JSON decoding"
        
        print(parsed_response)
        # Save structured information into a JSON file
        json_output_path = "./all_output/" + doi + ".json"
        
        with open(json_output_path, "w", encoding="utf-8") as json_file:
            json.dump(parsed_response, json_file, indent=4)


        authors = ', '.join(parsed_response['Authors'])
        keywords = ', '.join(parsed_response['Keywords'])
        relevant_fields = ', '.join(parsed_response.get('Relevant fields', []))  # Handles optional fields gracefully

        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as csvfile:
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

def change_vpn(current_index: int) -> int:
    """Change VPN location.
    
    Args:
        vpn_locations (List[str]): List of VPN locations.
        current_index (int): Index of the current VPN location.

    Returns:
        int: Index of the next VPN location.
    
    Raises:
        Exception: If all VPN locations fail.
    """
    return True
    # for i in range(len(vpn_locations)):
    #     try:
    #         location = vpn_locations[i]
    #         command = "nordvpn connect " + location
    #         result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    #         if result.returncode == 0:
    #             print("Connection successful.")
    #         else:
    #             print("Connection failed.")
    #             print("Error output:")
    #             print(result.stderr)
    #         return (current_index + i + 1) % len(vpn_locations)
    #     except subprocess.CalledProcessError:
    #         continue
    # raise Exception("All VPN locations failed")

def download_pdf(url: str, folder: str, journal_name: str, article_link: str) -> bool:
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

        convert_pdf_to_text(filepath)
        return True
    else:
        print(f'Error {response.status_code} while downloading {url}')
        return False
    
# **************************************************************** 
# **************************************************************** 
# **************************************************************** SCRAPERS

def scrape_page_articles_springer(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """Scrape articles from a webpage and download PDFs.
    
    Args:
        url (str): The URL of the webpage.
        count (int): The current count of downloaded PDFs.
        vpn_index (int): The index of the current VPN location.
        vpn_locations (List[str]): List of VPN locations.

    Returns:
        Tuple[Optional[str], int, int]: The URL of the next page, updated count, and updated VPN index.
    """
    count = 0

    page = requests.get(url + "/articles")
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        pdf_links = soup.find_all('a', attrs={'data-test': 'pdf-link'})
        base_url = url
        with ThreadPoolExecutor(max_workers=5) as executor:
            for link in pdf_links:
                print("="*50)
                future = executor.submit(download_pdf, base_url + link['href'], OUTPUT_FOLDER, journal_name, base_url + link['href'])
                count += 1
                if future.result():
                    if count % 10 == 0:
                        vpn_index = change_vpn(vpn_index)
        next_page_link = soup.find('a', attrs={'data-test': 'next-page'})
        if next_page_link:
            return base_url + next_page_link['href'], count, vpn_index
        else:
            return None, count, vpn_index
    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
        return None, count, vpn_index

def scrape_page_articles_rsc(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """
    Scrape articles from an RSC journal webpage, download PDFs, and navigate to the next page.

    Args:
        url (str): The URL of the webpage.
        journal_name (str): The name of the journal for logging purposes.
        count (int): The current count of downloaded PDFs.
        vpn_index (int): The index of the current VPN location.
        vpn_locations (List[str]): List of VPN locations.

    Returns:
        Tuple[Optional[str], int, int]: The URL of the next page, updated count, and updated VPN index.
    """
    count = 0

    page = requests.get(url)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        pdf_links = soup.find_all('a', class_='btn btn--primary btn--tiny', href=True)
        next_page_btn = soup.find('a', class_='paging__btn paging__btn--next', aria_disabled="false")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for link in pdf_links:
                # Construct the full URL for the PDF download link
                pdf_url = url + link['href']
                futures.append(executor.submit(download_pdf, url, OUTPUT_FOLDER, journal_name, pdf_url))

            for future in futures:
                if future.result():
                    count += 1
                    if count % 10 == 0:
                        vpn_index = change_vpn(vpn_index)

        if next_page_btn and next_page_btn.has_attr('data-pageno'):
            # Assuming the base URL remains constant, adjust the method to increment page numbers or change URL as needed
            next_page_no = next_page_btn['data-pageno']
            next_page_url = f"{url.split('#')[0]}#!recentarticles&adv&page={next_page_no}"
            return next_page_url, count, vpn_index
        else:
            return None, count, vpn_index
    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
        return None, count, vpn_index
    
def scrape_page_articles_acs(url: str, journal_name: str, base_url: str, vpn_index: int) -> Tuple[int, int]:
    """
    Scrape open access articles from an ACS, navigate to PDF page, and download PDFs.

    Args:
        url (str): The URL of the webpage.
        journal_name (str): The name of the journal for logging purposes.
        vpn_index (int): The index of the current VPN location.
        vpn_locations (List[str]): List of VPN locations.
        base_url (str): The base URL of the journal for constructing full links.

    Returns:
        Tuple[int, int]: Updated count of downloaded PDFs and updated VPN index.
    """
    count = 0

    page = requests.get(url)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.select('.issue-item_footer')

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for article in articles:
                open_access_img = article.find('img', alt="Open Access")
                if open_access_img:
                    pdf_link_element = article.find('a', title="PDF")
                    if pdf_link_element:
                        pdf_page_url = base_url + pdf_link_element['href']
                        # Navigate to the PDF page to find the actual PDF download link
                        pdf_page = requests.get(pdf_page_url)
                        if pdf_page.status_code == 200:
                            pdf_soup = BeautifulSoup(pdf_page.content, 'html.parser')
                            download_button = pdf_soup.find('a', class_='navbar-download')
                            if download_button and 'href' in download_button.attrs:
                                final_pdf_url = base_url + download_button['href']
                                futures.append(executor.submit(download_pdf, final_pdf_url, OUTPUT_FOLDER, journal_name, final_pdf_url))

            for future in futures:
                if future.result():
                    count += 1
                    print(f"Downloaded {count} PDFs")
                    if count % 10 == 0:
                        vpn_index = change_vpn(vpn_index)

    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")

    return count, vpn_index
    
def scrape_page_articles_nature(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """Scrape articles from Nature's website and download PDFs of open-access articles using concurrent futures for parallelization."""

    current_url = url
    count = 0

    while current_url:
        response = requests.get(current_url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            break

        soup = BeautifulSoup(response.content, 'html.parser')
        article_items = soup.select('li.app-article-list-row__item')

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for article_item in article_items:
                if article_item.find('span', class_='u-color-open-access'):
                    pdf_link_element = article_item.find('a', {'data-article-pdf': 'true', 'data-test': 'download-pdf'})
                    if pdf_link_element:
                        pdf_url = url + pdf_link_element['href']
                        futures.append(executor.submit(download_pdf, pdf_url, OUTPUT_FOLDER, journal_name, current_url))

            for future in futures:
                if future.result():
                    count += 1
                    print(f"Downloaded {count} PDFs")
                    if count % 10 == 0:
                        vpn_index = change_vpn(vpn_index)

        # Pagination
        next_page_link = soup.find('a', class_='c-pagination__link')
        current_url = url + next_page_link['href'] if next_page_link and 'href' in next_page_link.attrs else None

    print(f"Finished downloading. Total PDFs downloaded: {count}")
    return None, count, vpn_index

def scrape_page_articles_peerj(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """Scrape open access articles from PeerJ and download PDFs.

    Args:
        url (str): The URL of the webpage to start scraping from.
        journal_name (str): The name of the journal for logging and CSV updates.
        vpn_index (int): The index of the current VPN location.
        vpn_locations (List[str]): List of VPN locations.

    Returns:
        Tuple[Optional[str], int, int]: The URL of the next page (if any), updated count of downloaded PDFs, and updated VPN index.
    """
    count = 0

    page = requests.get(url)
    if page.status_code == 200:
        soup = BeautifulSoup(page.content, 'html.parser')
        articles = soup.select('div.main-search-item-row')

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for article in articles:
                article_link_element = article.find('a', href=True)
                if article_link_element:
                    article_url = f"{url.rsplit('/', 1)[0]}{article_link_element['href']}"
                    futures.append(executor.submit(download_pdf, article_url, OUTPUT_FOLDER, journal_name, url))

            for future in futures:
                result = future.result()
                if result:
                    count += 1
                    if count % 10 == 0:
                        vpn_index = change_vpn(vpn_index)

        # Identify the button for the next page and construct its URL
        next_page_button = soup.find('button', {'aria-label': 'Next page'})
        if next_page_button:
            # This assumes the next page URL needs to be extracted or constructed similarly
            next_page_url = None  # Adjust this logic to extract or construct the next page URL
            return next_page_url, count, vpn_index
        else:
            return None, count, vpn_index
    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
        return None, count, vpn_index
    
def scrape_page_articles_aiche(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """Scrape articles from AICHE using concurrent futures for parallel downloads."""
    current_url = url
    count = 0
    while current_url:
        response = requests.get(current_url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.select('li.search__item')

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for article in articles:
                    open_access = article.find('div', class_='open-access')
                    if open_access:
                        title_link = article.find('a', href=True)
                        if title_link:
                            article_url = url + title_link['href']
                            pdf_link_element = article.find('a', class_='pdf-download', href=True)
                            if pdf_link_element:
                                pdf_url = url + pdf_link_element['href'].replace('/epdf/', '/pdfdirect/') + "?download=true"
                                futures.append(executor.submit(download_pdf, pdf_url, OUTPUT_FOLDER, journal_name, url))

                for future in futures:
                    if future.result():
                        count += 1
                        if count % 10 == 0:
                            vpn_index = change_vpn(vpn_index)

            # Pagination
            next_page_link = soup.find('a', class_='pagination__next', href=True)
            current_url = url + next_page_link['href'] if next_page_link else None
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            break

    return None, count, vpn_index

def scrape_page_articles_wiley(url: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
    """Scrape open access articles from a Wiley journal webpage and download PDFs using concurrent futures for parallelization."""
    count = 0
    while url:
        page = requests.get(url)
        if page.status_code == 200:
            soup = BeautifulSoup(page.content, 'html.parser')
            articles = soup.select('li.search__item')

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for article in articles:
                    pdf_link_element = article.find('a', href=True, text=re.compile("PDF"))
                    if pdf_link_element:
                        pdf_url = "https://chemistry-europe.onlinelibrary.wiley.com" + pdf_link_element['href']
                        futures.append(executor.submit(download_pdf, pdf_url, OUTPUT_FOLDER, journal_name, url))

                for future in futures:
                    if future.result():
                        count += 1
                        if count % 10 == 0:
                            vpn_index = change_vpn(vpn_index)

            # Pagination
            next_page_link = soup.find('a', class_='pagination__btn--next', href=True)
            url = next_page_link['href'] if next_page_link else None
        else:
            print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
            break

    return None, count, vpn_index

# **************************************************************** 
# **************************************************************** 
# **************************************************************** SCRAPERS

def main():
    """Main function to initiate scraping and downloading."""

    publishers_links = {
        "springer_links" : ['https://jast-journal.springeropen.com', 'https://bioresourcesbioprocessing.springeropen.com', 'https://applbiolchem.springeropen.com', 'https://ijmme.springeropen.com', 'https://mnsl-journal.springeropen.com', 'https://chembioagro.springeropen.com', 'https://functionalcompositematerials.springeropen.com', 'https://materialstheory.springeropen.com', 'https://nanoconvergencejournal.springeropen.com', 'https://ejnmmipharmchem.springeropen.com', 'https://fjps.springeropen.com/'], # From at least 2014 up to and including 2024

        "rsc_links" : ['https://pubs.rsc.org/en/journals/journalissues/sc#!recentarticles&adv', 'https://pubs.rsc.org/en/journals/journalissues/cc?&_ga=2.70573268.533760951.1534757065-1046491195.1532073717#!recentarticles&adv'], # All 2024

        "acs_links" : ['https://pubs.acs.org/toc/abmcb8/current#', 'https://pubs.acs.org/toc/accacs/0/0', 'https://pubs.acs.org/toc/aoiab5/0/0', 'https://pubs.acs.org/toc/bcches/0/0', 'https://pubs.acs.org/toc/inocaj/0/0', 'https://pubs.acs.org/toc/joceah/0/0', 'https://pubs.acs.org/toc/jacsat/0/0', 'https://pubs.acs.org/toc/mpohbp/0/0', 'https://pubs.acs.org/toc/oprdfk/0/0', 'https://pubs.acs.org/toc/orgnd7/0/0', 'https://pubs.acs.org/toc/pcrhej/0/0', 'https://pubs.acs.org/toc/cbehb5/0/0'], # Mostly 2024/2023
        
        "nature_links" : ['https://www.nature.com/commschem/research-articles'], # 2018 up to and including 2024

        "peerj_links" : ['https://peerj.com/articles/?section=microbiology', 'https://peerj.com/articles/?section=biochemistry-biophysics-molecular-biology', 'https://peerj.com/articles/?journal=ochem', 'https://peerj.com/articles/?journal=ichem'], 

        "aiche_links" : ['https://aiche.onlinelibrary.wiley.com/action/doSearch?SeriesKey=23806761&sortBy=Earliest'],

        "wiley_links" : ['https://chemistry-europe.onlinelibrary.wiley.com/action/doSearch?SeriesKey=27514765&sortBy=Earliest', 'https://chemistry-europe.onlinelibrary.wiley.com/action/doSearch?SeriesKey=21911363&sortBy=Earliest']
    }

    # ================================================================================================== 

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Ensure the CSV file exists
    fieldnames = [
    'DOI', 'Title', 'Abstract', 'Journal', 'Relevant fields', 
    'Authors', 'Keywords', 'Institute of Origin', 'Funding', 
    'Methods', 'Results', 'Experiment details'
    ]

    # Check if the CSV file exists
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    vpn_index = 0
    # Iterate over the publishers and their links
    for publisher_key, links in publishers_links.items():
        # Extract the journal name from the publisher_key
        journal_name = publisher_key.replace("_links", "")
        print(f"Starting scraping for {journal_name}")

        # Build the function name dynamically
        function_name = f"scrape_page_articles_{journal_name}"

        # Check if this function exists in the global namespace
        scraper_function = globals().get(function_name)

        if scraper_function:
            for link in links:
                print(f"Scraping {link}")
                try:
                    # url: str, journal_name: str, count: int, vpn_index: int
                    # Call the dynamically selected scraper function
                    url, temp_count, vpn_index = scraper_function(link, journal_name, vpn_index)
                    print(f"Completed scraping for {link}")
                except Exception as e:
                    print(f"Error occurred while scraping {link}: {e}")
                    traceback.print_exc()
                print("="*50)  # Divider after each link
            print("="*100)  # Divider after each publisher
        else:
            print(f"No scraping function found for {journal_name}")


if __name__ == '__main__':
    main()