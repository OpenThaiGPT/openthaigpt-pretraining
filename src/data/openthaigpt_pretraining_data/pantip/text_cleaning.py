from bs4 import BeautifulSoup


def clean_html_tags(data: str):
    """
    Description:
        Remove HTML tags using BeautifulSoup.
    Args:
        data: Input string.
    Returns:
        clean_data: Text output without HTML tags.
    """
    soup = BeautifulSoup(data, "html.parser")
    cleaned_tags = soup.get_text(separator=" ")

    return cleaned_tags
