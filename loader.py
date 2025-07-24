import os
from langchain_core.documents import Document
from bs4 import BeautifulSoup
import requests

os.environ['USER_AGENT'] = 'myagent'
headers = {"User-Agent": os.environ['USER_AGENT']}


# clean_docs will remove the header, footer, and table of content of the sources
def clean_docs(url, headers):
    try: 
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        html = response.text 

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
            tag.decompose()

        toc_classes = ["toc", "toctree-wrapper", "wy-nav-side", "rst-content-toc"]
        for class_name in toc_classes:
            for toc_tag in soup.find_all(class_=class_name):
                toc_tag.decompose()
        
        raw_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)
        

        return Document(page_content=clean_text, metadata={"source": url}) 

    except Exception as e:
        print(f"\n Failed to process {url}: {e}" )



# loader_docs will load all the docuemnts to "docs"
def loader_docs():
    urls = ["https://chameleoncloud.readthedocs.io/en/latest/#what-is-chameleon",
                            "https://chameleoncloud.readthedocs.io/en/latest/getting-started/index.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/federation.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/pi_eligibility.html",
                            "https://chameleoncloud.readthedocs.io/en/latest/user/project.html",
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
                            "https://python-chi.readthedocs.io/en/latest/"]

    docs = []
    for url in urls:
        doc = clean_docs(url, headers)
        if doc:
            docs.append(doc)
    return docs

if __name__ == "__main__":
    docs = loader_docs()


