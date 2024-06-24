import traceback
import os
import csv
from .publishers_links import publishers_links

OUTPUT_FOLDER: str = "../data/all_output"
CSV_PATH: str = "../data/downloaded_articles.csv"

def main():
    """Main function to initiate scraping and downloading of open access papers."""

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    fieldnames = [
    'DOI', 'Title', 'Abstract', 'Journal', 'Relevant fields', 
    'Authors', 'Keywords', 'Institute of Origin', 'Funding', 
    'Methods', 'Results', 'Experiment details'
    ]

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    vpn_index = 0
    for publisher_key, links in publishers_links.items():
        journal_name = publisher_key.replace("_links", "")
        print(f"Starting scraping for {journal_name}")
        function_name = f"scrape_page_articles_{journal_name}"
        scraper_function = globals().get(function_name)

        if scraper_function:
            for link in links:
                print(f"Scraping {link}")
                try:
                    # url: str, journal_name: str, count: int, vpn_index: int
                    # Call the dynamically selected scraper function
                    url, temp_count, vpn_index = scraper_function(link, OUTPUT_FOLDER, CSV_PATH, journal_name, vpn_index)
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