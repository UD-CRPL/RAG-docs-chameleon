import os
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import requests


os.environ['USER_AGENT'] = 'myagent'
headers = {"User-Agent": os.environ['USER_AGENT']}

#Fetch the HTML Document
urls = ["https://chameleoncloud.readthedocs.io/en/latest/",
                            " https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/federation.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/pi_eligibility.html",
                            " https://chameleoncloud.readthedocs.io/en/latest/user/project.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/profile.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/help.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/gui/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/cli/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/jupyter/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/discovery/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/reservations/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/baremetal/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/images/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/power_monitoring/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/swift/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/shares/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/fpga/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/sharing/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/daypass/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/technical/kvm/index.html",
                            "https://chameleoncloud.org/learn/frequently-asked-questions/",
                            "https://chameleoncloud.org/blog/",
                            "https://chameleoncloud.org/blog/2025/07/01/chameleon-changelog-for-june-2025/",
                            "https://chameleoncloud.org/blog/2025/06/25/infrastructure-without-scaling-limits/",
                            "https://chameleoncloud.org/blog/2025/06/20/accelerate-your-research-with-nvidia-h100-gpus-on-kvmtacc",
                            "https://chameleoncloud.org/blog/2025/06/02/chameleon-changelog-for-may-2025/",
                            "https://chameleoncloud.org/blog/2025/05/27/teaching-cloud-computing-with-chameleon-making-complex-concepts-accessible/",
                            "https://chameleoncloud.org/blog/2025/05/19/leveraging-new-and-improved-chameleon-images/",
                            "https://chameleoncloud.org/blog/2025/05/01/chameleon-changelog-for-april-2025/",
                            "https://chameleoncloud.org/blog/2025/05/01/repeto-releases-report-on-challenges-of-practical-reproducibility-for-systems-and-hpc-computer-science/",
                            "https://chameleoncloud.org/blog/2025/04/29/fair-co2-fair-attribution-for-cloud-carbon-emissions/",
                            "https://chameleoncloud.org/blog/2025/04/29/faster-multimodal-ai-lower-gpu-costs/",
                            "https://python-chi.readthedocs.io/en/latest/",
                            "https://forum.chameleoncloud.org/"]


'''
response = requests.get(url)

if response.status_code == 200:
    html_content = response.text
else: print("Failed to retirve the document")


soup = BeautifulSoup(html_content, 'html.parser')
#clean the header, footer, ...
for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
    tag.decompose()

#get the clean text

raw_text = soup.get_text(separator="\n")
#remove extra space and blank lines
lines = [line.strip() for line in raw_text.splitlines()]
clean_text = "\n".join(line for line in lines if line)

doc = Document(page_content=clean_text, metadata={"source": url})

print(doc.page_content)

'''


documents = []

for url in urls: 
    try: 
        response = requests.get(rul, headers=headers)
        response.raise_for_status()
        html = response.text 

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
            tag.decompose
        
        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)
        

        doc = Document(page_content=clean_text, metadata={"source": url})
        documents.append(doc)

        print(f"\n✅ Cleaned content from: {url}")
        print(doc.page_content[:500])
        print("-" * 80)

    except Exception as e:
        print(f"\n Failed to process {url}: {e}" )

