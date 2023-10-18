import requests
from typing import Callable
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from markdownify import markdownify as md

ALLOWED_CONTENT_TYPES = [
    "text/html",
    "application/json",
    "application/xml",
    "application/javascript",
    "text/plain",
]

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.5",
    "Alt-Used": "LEAVE-THIS-KEY-SET-BY-TOOL",
    "Connection": "keep-alive",
    "Host": "LEAVE-THIS-KEY-SET-BY-TOOL",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/111.0",
}


class WebFetcher:
    def __init__(self, config=None):
        self._config = {
            "htmlToMarkdown": True,
            "summarizeHtml": False,
        }
        if config:
            self._config.update(config)

    def fetch(self, uri: str) -> str:
        headers = DEFAULT_HEADERS.copy()
        parsed_uri = urlparse(uri)
        headers["Host"] = parsed_uri.hostname
        headers["Alt-Used"] = parsed_uri.hostname

        response = requests.get(uri, headers=headers, **self._config.get("requestConfig", {}))
        response.raise_for_status()

        content_type = response.headers["content-type"]
        content_type_array = content_type.split(";")
        if not content_type_array[0] or content_type_array[0] not in ALLOWED_CONTENT_TYPES:
            raise Exception(f"Site returned an invalid content type of {content_type}")

        doc_type = content_type_array[0].split("/")[1] if content_type_array[0] != "text/plain" else None
        if doc_type == "html" and self._config["htmlToMarkdown"]:
            text = self.html_to_markdown(response.text, uri)
            return text
        else:
            return response.text

    @staticmethod
    def html_to_markdown(html: str, base_url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")

        for script in soup.find_all("script"):
            script.extract()

        for a in soup.find_all("a"):
            href = a.get("href")
            if href and not href.startswith("http"):
                try:
                    a["href"] = requests.compat.urljoin(base_url, href)
                except ValueError:
                    pass

        markdown = md(str(soup.body), heading_style="ATX", bullet_style="-", code_style="backticks")
        markdown = "\n\n".join(markdown.splitlines())
        if len(markdown) > 64:
            start = markdown.find("\n")
            if start != -1:
                markdown = markdown[start:]
            else:
                start = markdown.find(" ")
                if start != -1:
                    markdown = markdown[start:]

        return markdown



# Example usage:
# web_fetcher = WebFetcher()
# # result = web_fetcher.fetch("https://earthshotprize.org/the-prize/earthshots/")
# result = web_fetcher.fetch("https://www.sec.gov/Archives/edgar/data/1442145/000143774923004945/vrsk20221231_10k.htm")
# print('test', result)

# # save to a test file
# with open("test3.md", "w") as f:
#     f.write(result)
