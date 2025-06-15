import os
import requests
import json
from typing import List
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown, display, update_display
from openai import OpenAI

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10:
    print("API key looks good so far")
else:
    print("There might be a problem with your API key? Please visit the troubleshooting notebook!")

MODEL = 'gpt-4o-mini'
openai = OpenAI()
headers = {
    "User-Agent": "Mozilla/5.0"
}

class Website:
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for tag in soup.body(["script", "style", "img", "input"]):
                tag.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""
        self.links = [link.get('href') for link in soup.find_all('a') if link.get('href')]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

link_system_prompt = """You are provided with a list of links found on a webpage. Respond in JSON as:
{
    "links": [
        {"type": "about page", "url": "https://full.url/here/about"},
        {"type": "careers page", "url": "https://another.url/careers"}
    ]
}
"""

def get_links_user_prompt(website):
    return f"Links from {website.url}:\n" + "\n".join(website.links)

def get_links(url):
    website = Website(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_user_prompt(website)}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def get_all_details(url):
    result = "Landing page:\n"
    result += Website(url).get_contents()
    links = get_links(url)
    print("Found links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += Website(link["url"]).get_contents()
    return result

system_prompt = "You are an assistant that builds brochures based on website info."

def get_brochure_user_prompt(company_name, url):
    return f"You are looking at {company_name}.\n{get_all_details(url)[:5000]}"

def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
        ]
    )
    display(Markdown(response.choices[0].message.content))

create_brochure("HuggingFace", "https://huggingface.co")
