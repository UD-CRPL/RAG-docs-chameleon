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
                            "https://python-chi.readthedocs.io/en/latest/",
                            "https://forum.chameleoncloud.org/t/openstack-python-client-issue-the-request-you-have-made-requires-authentication/88",
                            "https://forum.chameleoncloud.org/t/new-chameleon-images-less-setup-more-coding/124",
                            "https://forum.chameleoncloud.org/t/announcing-user-experiments-for-april-2025/89",
                            "https://forum.chameleoncloud.org/t/400-bad-request-to-keystone-with-chrome/86",
                            "https://forum.chameleoncloud.org/t/announcing-tips-and-tricks-for-april-2025/85",
                            "https://forum.chameleoncloud.org/t/kvm-and-port-5000-issues/84",
                            "https://forum.chameleoncloud.org/t/new-user-experiment-blog-mar-2025/83",
                            "https://forum.chameleoncloud.org/t/email-with-new-connection-from-client-ip/47",
                            "https://forum.chameleoncloud.org/t/date-not-showed/46",
                            "https://forum.chameleoncloud.org/t/recently-enrolled-raspberry-pi5-devices-dont-show-up-on-the-host-calendar/44",
                            "https://forum.chameleoncloud.org/t/enrolling-raspberry-pis-onto-chi-edge/43",
                            "https://forum.chameleoncloud.org/t/proposed-change-updating-cc-ubuntu24-04-kernel-from-6-8-to-6-11/41",
                            "https://forum.chameleoncloud.org/t/credentials-in-openrc-expired-need-to-refresh/34",
                            "https://forum.chameleoncloud.org/t/summer-of-reproducibility-call-for-projects/31",
                            "https://forum.chameleoncloud.org/t/fyi-about-ports-with-no-ip-address/25",
                            "https://forum.chameleoncloud.org/t/resolved-2025-02-07-kvm-tacc-issues-affecting-instance-launch/22",
                            "https://forum.chameleoncloud.org/t/querying-node-availability-in-python-chi/24"]

    docs = []
    for url in urls:
        doc = clean_docs(url, headers)
        if doc:
            docs.append(doc)
    return docs

if __name__ == "__main__":
    docs = loader_docs()


