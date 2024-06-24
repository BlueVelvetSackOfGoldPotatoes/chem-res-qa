import json
import csv
from io import StringIO

def filter_questions(data, include_keywords=None, exclude_keywords=None, fields=None, max_results=None, case_sensitive=False, output_format='json'):
    """
    Filter questions based on inclusion or exclusion of keywords in specified fields and limit the number of results.

    Parameters:
    - data (list): List of dictionaries containing the dataset.
    - include_keywords (list): List of keywords to include in the results.
    - exclude_keywords (list): List of keywords to exclude from the results.
    - fields (list or None): List of fields to search in. If None, searches all text fields.
    - max_results (int or None): Maximum number of questions to return. If None, returns all matching questions.
    - case_sensitive (bool): Boolean indicating if the search should be case sensitive.
    - output_format (str): The format of the output file ('json', 'csv', 'txt').

    Returns:
    - filtered_data (str): Filtered data in the specified output format.

    Use Case:
        Load dataset from URL:
        >>> data = load_dataset('https://raw.githubusercontent.com/BlueVelvetSackOfGoldPotatoes/chem-aca-q-a/main/data/chem_mqa_dataset.json')

        Filter questions and limit results:
        >>> filtered_data = filter_questions(
        ...     data,
        ...     include_keywords=['nanoparticle', 'uptake'],
        ...     exclude_keywords=['PLGA'],
        ...     fields=['Context', 'Question'],
        ...     max_results=10,  # Limit the results to 10 questions
        ...     case_sensitive=False,
        ...     output_format='csv'
        ... )

        Print or use filtered data:
        >>> print(filtered_data)
    """
    filtered_data = []
    for item in data:
        if max_results is not None and len(filtered_data) >= max_results:
            break  # Stop searching once the maximum number of results is reached
        for question_key, question_value in item.items():
            if isinstance(question_value, dict):
                content_to_search = {field: question_value.get(field, '') for field in (fields if fields else question_value.keys())}
                content_string = ' '.join(content_to_search.values())
                content_string = content_string if case_sensitive else content_string.lower()

                if include_keywords and not any((keyword if case_sensitive else keyword.lower()) in content_string for keyword in include_keywords):
                    continue  # Skip this item if no include keywords are found
                if exclude_keywords and any((keyword if case_sensitive else keyword.lower()) in content_string for keyword in exclude_keywords):
                    continue  # Skip this item if any exclude keywords are found

                filtered_data.append(item)
                break  # Break to avoid duplicating the same item if multiple fields match

    return format_output(filtered_data, output_format)

def format_output(data, format_type):
    """Formats the output data according to the specified format."""
    if format_type == 'json':
        return json.dumps(data, indent=4)
    elif format_type == 'csv':
        return data_to_csv(data)
    elif format_type == 'txt':
        return data_to_txt(data)
    else:
        raise ValueError("Unsupported format specified.")
        
def data_to_csv(data):
    """Converts dictionary data to a CSV string."""
    output = StringIO()
    if data:
        keys = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=keys)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return output.getvalue()

def data_to_txt(data):
    """Converts dictionary data to a plain text string, JSON formatted."""
    return '\n'.join(json.dumps(item, indent=4) for item in data)