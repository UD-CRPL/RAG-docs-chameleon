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
    urls = [
        "https://chameleoncloud.readthedocs.io/en/latest/technical/shares/mounting.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/shares/concepts.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/kvm/kvm_volumes.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/catalog.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/kvm/kvm_lbaas.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/sharing/packaging_artifacts.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/sharing.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/fpga/index.html#introduction",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/networks_vlan.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/swift/index.html#availability",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/images/gui_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/shares/gui_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/gui_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/heat_templates.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/networks_fabnet.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/cli_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/networks_jumbo_frames.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/power_monitoring/index.html#hardware-support",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/power_monitoring/index.html#getting-started",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/networks_stitching.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/sharing/browsing_artifacts.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/images/cli_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/kvm/kvm_gui.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/networks/networks_basic.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/kvm/kvm_instance_migration.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/swift/index.html#objects-and-containers",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/shares/cli_management.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/fpga/index.html#reserving-fpga-hardware",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/complex/advanced_topics.html",
        "https://chameleoncloud.readthedocs.io/en/latest/technical/power_monitoring/index.html#available-power-monitoring-methods",
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
        "https://chameleoncloud.org/blog/2024/08/19/composible-hardware-on-chameleon-now/",
        "https://chameleoncloud.org/blog/2025/04/21/importing-github-repositories-to-trovi-a-step-by-step-guide/",
        "https://chameleoncloud.org/blog/2024/06/18/power-measurement-and-management-on-chameleon/",
        "https://chameleoncloud.org/blog/2025/06/20/accelerate-your-research-with-nvidia-h100-gpus-on-kvmtacc/",
        "https://chameleoncloud.org/blog/2023/06/26/chameleon-images-overview/",
        "https://chameleoncloud.org/blog/2025/07/01/chameleon-changelog-for-june-2025/",
        "https://chameleoncloud.org/blog/2024/11/18/building-mpi-clusters-on-chameleon-a-practical-guide/",
        "https://chameleoncloud.org/blog/2023/05/01/how-to-port-your-experiments-between-chameleon-sites/",
        "https://chameleoncloud.org/blog/2024/11/01/chameleon-changelog-for-october-2024/",
        "https://chameleoncloud.org/blog/2023/10/02/chameleon-changelog-for-september-2023/",
        "https://chameleoncloud.org/blog/2025/03/17/extending-your-research-artifacts-lifespan/",
        "https://chameleoncloud.org/blog/2024/10/21/packaging-your-experiments-on-chameleon-with-python-chi-10/",
        "https://chameleoncloud.org/blog/2024/07/15/expanding-horizons-with-chiedge-new-peripheral-support/",
        "https://chameleoncloud.org/blog/2024/01/04/chameleon-changelog-for-december-2023/",
        "https://chameleoncloud.org/blog/2023/07/25/using-terraform-with-chameleon/",
        "https://chameleoncloud.org/blog/2023/11/01/chameleon-changelog-for-october-2023/",
        "https://chameleoncloud.org/blog/2023/03/20/the-practical-reproducibility-opportunity/",
        "https://chameleoncloud.org/blog/2022/12/12/tickets-of-the-year-2022/",
        "https://chameleoncloud.org/blog/2023/09/01/chameleon-changelog-for-august-2023/",
        "https://chameleoncloud.org/blog/2024/09/16/back-to-school-with-chameleon/",
        "https://chameleoncloud.org/blog/2025/01/20/top-read-blogs-on-chameleon-this-year-what-blogs-were-most-helpful-to-our-community/",
        "https://chameleoncloud.org/blog/2024/07/01/chameleon-changelog-for-june-2024/",
        "https://chameleoncloud.org/blog/2025/06/02/chameleon-changelog-for-may-2025/",
        "https://chameleoncloud.org/blog/2024/09/03/chameleon-changelog-for-august-2024/",
        "https://chameleoncloud.org/blog/2023/12/19/tickets-of-the-year-on-chameleon-2023/",
        "https://chameleoncloud.org/blog/2024/10/01/chameleon-changelog-for-september-2024/",
        "https://chameleoncloud.org/blog/2024/06/04/chameleon-changelog-for-may-2024/",
        "https://chameleoncloud.org/blog/2024/03/01/chameleon-changelog-for-february-2024/",
        "https://chameleoncloud.org/blog/2024/12/18/chameleon-tickets-of-the-year-2024/",
        "https://chameleoncloud.org/blog/2023/01/23/experiment-pattern-bastion-host/",
        "https://chameleoncloud.org/blog/2024/08/02/chameleon-changelog-for-july-2024/",
        "https://chameleoncloud.org/blog/2025/03/03/chameleon-changelog-for-february-2025/",
        "https://chameleoncloud.org/blog/2025/05/19/leveraging-new-and-improved-chameleon-images/",
        "https://chameleoncloud.org/blog/2023/12/01/chameleon-changelog-for/",
        "https://chameleoncloud.org/blog/2025/02/04/chameleon-changelog-for-january-2025/",
        "https://chameleoncloud.org/blog/2024/04/15/chiedge-transitioning-from-successful-preview-to-full-production/",
        "https://chameleoncloud.org/blog/2025/01/02/chameleon-changelog-for-december-2024/",
        "https://chameleoncloud.org/blog/2023/08/29/running-experiments-inside-a-jupyter-notebook/",
        "https://chameleoncloud.org/blog/2024/02/01/chameleon-changelog-for-january-2023/",
        "https://chameleoncloud.org/blog/2024/05/21/seamless-ssh-container-access-with-chiedge/",
        "https://chameleoncloud.org/blog/2024/05/01/chameleon-changelog-for-april-2024/",
        "https://chameleoncloud.org/blog/2024/04/01/chameleon-changelog-for-march-2024/",
        "https://chameleoncloud.org/blog/2025/07/21/the-hitchhikers-guide-to-chameleon-documentation-finding-answers-fast/",
        "https://chameleoncloud.org/blog/2025/05/01/chameleon-changelog-for-april-2025/",
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
        "https://forum.chameleoncloud.org/t/querying-node-availability-in-python-chi/24"
    ]

    docs = []
    for url in urls:
        doc = clean_docs(url, headers)
        if doc:
            docs.append(doc)
    return docs

if __name__ == "__main__":
    docs = loader_docs()


