import requests
import re
from typing import Tuple, Optional
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from .pdf_utils import download_pdf

def scrape_page_articles_springer(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                future = executor.submit(download_pdf, base_url + link['href'], output_folder, csv_path, journal_name, base_url + link['href'])
                count += 1
                
        next_page_link = soup.find('a', attrs={'data-test': 'next-page'})
        if next_page_link:
            return base_url + next_page_link['href'], count, vpn_index
        else:
            return None, count, vpn_index
    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
        return None, count, vpn_index

def scrape_page_articles_rsc(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                futures.append(executor.submit(download_pdf, url, output_folder, csv_path, journal_name, pdf_url))

            for future in futures:
                if future.result():
                    count += 1

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
    
def scrape_page_articles_acs(url: str, output_folder: str, csv_path:str, journal_name: str, base_url: str, vpn_index: int) -> Tuple[int, int]:
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
                                futures.append(executor.submit(download_pdf, final_pdf_url, output_folder, csv_path, journal_name, final_pdf_url))

            for future in futures:
                if future.result():
                    count += 1
                    print(f"Downloaded {count} PDFs")

    else:
        print(f"Failed to retrieve the webpage. Status code: {page.status_code}")

    return count, vpn_index
    
def scrape_page_articles_nature(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                        futures.append(executor.submit(download_pdf, pdf_url, output_folder, csv_path, journal_name, current_url))

            for future in futures:
                if future.result():
                    count += 1
                    print(f"Downloaded {count} PDFs")

        # Pagination
        next_page_link = soup.find('a', class_='c-pagination__link')
        current_url = url + next_page_link['href'] if next_page_link and 'href' in next_page_link.attrs else None

    print(f"Finished downloading. Total PDFs downloaded: {count}")
    return None, count, vpn_index

def scrape_page_articles_peerj(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                    futures.append(executor.submit(download_pdf, article_url, output_folder, csv_path, journal_name, url))

            for future in futures:
                result = future.result()
                if result:
                    count += 1

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
    
def scrape_page_articles_aiche(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                                futures.append(executor.submit(download_pdf, pdf_url, output_folder, csv_path, journal_name, url))

                for future in futures:
                    if future.result():
                        count += 1

            # Pagination
            next_page_link = soup.find('a', class_='pagination__next', href=True)
            current_url = url + next_page_link['href'] if next_page_link else None
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            break

    return None, count, vpn_index

def scrape_page_articles_wiley(url: str, output_folder: str, csv_path: str, journal_name: str, vpn_index: int) -> Tuple[Optional[str], int, int]:
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
                        futures.append(executor.submit(download_pdf, pdf_url, output_folder, csv_path, journal_name, url))

                for future in futures:
                    if future.result():
                        count += 1

            # Pagination
            next_page_link = soup.find('a', class_='pagination__btn--next', href=True)
            url = next_page_link['href'] if next_page_link else None
        else:
            print(f"Failed to retrieve the webpage. Status code: {page.status_code}")
            break

    return None, count, vpn_index