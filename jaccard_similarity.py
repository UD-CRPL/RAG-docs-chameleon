from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def jaccard_similarity_func(text1, text2):
    
    #create sets of words
    txt1 = text1.lower().replace('.','').replace(',','')
    txt2 = text2.lower().replace('.','').replace(',','')


    #tokenizing and creating binary vectors
    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform([txt1, txt2]).toarray()

    #create sets of indecies 
    set1 = set(np.where(vectors[0] > 0)[0])
    set2 = set(np.where(vectors[1] > 0)[0])
        
    #intersection
    common_words = set1.intersection(set2)

    #create union set
    all_words = set1.union(set2)

    #return 1 when both texts are empty
    if not all_words:
        return 1.0
    return len(common_words)/len(all_words)


'''ground_truth = "cat dog mouse"
rag_answer = "cat dog elephant"
#common words = 2
#All unique words = 4
#Expected Jaccard similarity= 2/4 = 0.5

score1 = jaccard_similarity_func(ground_truth, rag_answer)


ground_truth2 = "zahra is pretty."
rag_answer2 = "saieda is pretty and smart."
#common words = 2
#All unique words = 6
#Expected Jaccard similarity= 2/6 = 0.5

score2 = jaccard_similarity_func(ground_truth2, rag_answer2)



print(f"similarity of rag2 answer in compare to ground truth is: {score2:.4f}")
print(f"similarity of rag answer in compare to ground truth is: {score1:.4f}")'''


def evaluate_score(ground_truth, rag_answer):
    scores = []
    for i in range(len(ground_truth)):
        GT = ground_truth[i]
        RA = rag_answer[i]
        score = jaccard_similarity_func(GT,RA)
        scores.append(score)
        print(f"query{i+1} similarity: {score:.4f}")
    return scores
    


ground_truth=[  "Associate Sites are smaller-scale, independently operated racks connected to the Chameleon core infrastructure, allowing other institutions to contribute unique hardware resources to the testbed.", 
        "The gpu_rtx_8000 nodes are equipped with NVIDIA Quadro RTX 8000 GPUs, each with 48GB of memory.",
        "In the Chameleon Portal, navigate to the 'Network' section and click 'Networks'. Select 'Create Network', give it a name, and then create a subnet within it, specifying an address range (e.g., 192.168.1.0/24).",
        "Log in to the Chameleon JupyterHub. In the launcher, you can browse pre-made notebooks under 'Notebooks' or 'Tutorials' which cover topics like basic Chameleon usage, machine learning, and networking.",
        "This can sometimes happen. The first step is to check the node's remote console from the Chameleon UI for any error messages during boot. If there are no clear errors, try deleting the instance and launching it again. If the problem persists, you should open a support ticket.",
        "This typically means your openrc.sh file is not sourced correctly or has expired credentials. Re-download your openrc.sh v3 file from the Chameleon UI, source it again with source <your-openrc-file.sh>, and enter your password when prompted.",
        "CHI-in-a-Box is a packaged version of the Chameleon infrastructure (CHI) that allows other institutions to build their own Chameleon-like private cloud, capable of federating with the main testbed.",
        "Deep reconfigurability means researchers can go beyond virtual machines and get bare metal access to hardware. This allows them to interact with and modify low-level systems like the operating system kernel, networking configurations, and even firmware on some hardware.",
        "A Private IP is an internal address used for communication between instances on the same private network. A Floating IP is a public, internet-routable IP address that you can associate with an instance to make it reachable from the outside world.",
        "Use a GPU node for massively parallel workloads like machine learning, scientific simulations, or data analytics. Use an FPGA (Field-Programmable Gate Array) node when your experiment requires designing and testing custom digital logic circuits or hardware accelerators.",
        "You should cite the primary paper: 'Chameleon: A Large-Scale, Reconfigurable Experimental Environment for Cloud Research. Jason Anderson, et al.' and include your project ID. The specific BibTeX entry is available in the documentation.",
        "First, create a volume under the 'Volumes' tab in the UI. Then, while the volume is unattached, find your running instance, select 'Attach Volume' from its dropdown menu, and choose the volume you created. The system will then attach it as a new block device (e.g., /dev/vdb).",
        "The maximum length for a single lease is typically 7 days. For longer experiments, you need to chain multiple reservations or contact support for special arrangements.",
        "No. The Chameleon Acceptable Use Policy strictly prohibits using the testbed for any commercial activity, including cryptocurrency mining. Allocations are for open computer science research and education only.",
        "You can create a custom disk image. Start by launching a base OS image on a bare metal node, install the desired kernel and any other software, and then create a snapshot of the node's disk. This snapshot image can then be used to launch other nodes with your custom configuration.",
        "Trovi is Chameleon's artifact sharing portal. It allows users to package, share, and reproduce entire experiments, including Jupyter notebooks, disk images, and orchestration templates, making research more discoverable and repeatable.",
        "You write a YAML-based Heat Orchestration Template (HOT) that defines all the resources (servers, networks, volumes). Then you launch the entire stack from the GUI under 'Orchestration' -> 'Stacks' or via the CLI with openstack stack create.",
        "The two core sites for the Chameleon testbed are at the University of Chicago (UC) and the Texas Advanced Computing Center (TACC).The two core sites for the Chameleon testbed are at the University of Chicago (UC) and the Texas Advanced Computing Center (TACC).",
        "Check the security group rules associated with your instance. Ensure there is an 'egress' rule that allows outbound traffic. The default security group usually allows all egress, but a custom one might not.",
        "Find the experiment on the Trovi artifact repository, which is often linked from the blog post. Import the artifact into your Chameleon account. This will typically give you the Jupyter notebook, scripts, and environment needed to re-run the experiment with a few clicks."]

rag_answer=["I don't know. The context provided does not mention 'Chameleon Associate Sites'. However, it mentions various features and capabilities of the Chameleon testbed, including resource isolation, custom images, power monitoring, performance metrics, advanced networking, collaboration & reproducibility, and more. If you are looking for information on how to access or use a specific feature related to associate sites, I would recommend checking the FAQs site: https://docs.chameleoncloud.org/en/latest/faq.html",
            "The specific model of GPU available on the gpu_rtx_8000 nodes is not explicitly mentioned in the provided context. However, it can be inferred that these nodes are equipped with RTX 6000 GPUs since they are listed as 'Supported (with either open source or proprietary driver)' in the section about GPU support status. Source: The relevant information comes from the GPU support status section of the documentation, which is linked here: https://docs.chameleoncloud.org/en/latest/gpu_support_status.html",
            "To create an isolated network for your instances using the GUI, follow these steps: Go to the Networking section of the Chameleon dashboard. Click on 'Create Isolated Network' and select the VLAN ID for your isolated network. Configure the network settings as needed (e.g., IP address range, subnet mask). Click 'Create' to create the isolated network. This information comes from the Networking documentation page: https://docs.chameleoncloud.org/en/latest/networking.html#isolated-networks",
            "To find a pre-configured Jupyter notebook on Chameleon, you can go to the Chameleon portal and access the JupyterHub interface. From there, you can launch a Jupyter server with an environment pre-configured with python-chi and authentication to the testbed. This information comes from the 'Jupyter on Chameleon' section of the documentation: https://docs.chameleoncloud.org/en/latest/user/jupyter.html",
            "Your bare metal node is stuck in the 'deploying' state for a long time. To resolve this issue, please check if you have made a reservation for the node and if it has been successfully created. You can verify this by checking your Leases page on Chameleon. If the lease status is still 'PENDING', try waiting for some time or contact our Help Desk for further assistance. This information comes from the FAQs site: https://chameleoncloud.org/faq/.",
            "The error 'Error 403: Forbidden' when using the OpenStack CLI is not mentioned in the provided context. However, a similar issue with authentication was reported by user mcanonic on May 19, 2025, where they received an HTTP 401 error (The request you have made requires authentication). This issue was later resolved without any changes being made. This information comes from the forum discussion and here is the link to the site: https://chameleoncloud.org/forum/",
            "What is 'CHI-in-a-Box'? CHI-in-a-box is a packaging of the implementation of the core services that together constitute the Chameleon testbed for experimental Computer Science research. It allows users to discover information about resources, allocate them, configure them, and monitor various types of metrics. This information comes from the FAQs site: https://www.chameleoncloud.org/docs/faqs.html",
            "According to the context, 'deep reconfigurability' refers to Chameleon's ability to provide researchers with deeply reconfigurable cloud infrastructure for systems, networking, distributed computing, and security research. This means that users can customize their experimental setup by requesting specific hardware configurations, which will be fulfilled by the Chameleon team. This information comes from the Key Features section of the documentation, specifically under 'Experimental Capabilities'. Source: https://chameleoncloud.org/docs/",
            "A Floating IP is a publicly accessible IP address associated with an instance in Chameleon Cloud, allowing it to be accessed from outside the testbed. On the other hand, a Private IP is used for communication between instances within the same project and site. This information comes from the documentation on 'Networks' and here is the link to the site: https://docs.chameleoncloud.org/en/latest/user/issues.html#networks",
            "You should use an FPGA node versus a GPU node when you need high-performance computing capabilities for tasks such as data processing, machine learning, or scientific simulations that require custom hardware acceleration. This information comes from the FPGAs section of the Chameleon Cloud Documentation and here is the link to the site: https://docs.chameleoncloud.org/en/latest/fpgas/index.html",
            "To cite Chameleon in your research paper, you should use the following reference: Keahey, K., Anderson, J., Zhen, Z., Riteau, P., Ruth, P., Stanzione, D., ... & Stubbs, J. (2020). Lessons Learned from the Chameleon Testbed. In Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC '20) (pp. 1-12). USENIX Association. You can also use the BibTex entry provided: @incollection{keahey2020lessons, title={Lessons Learned from the Chameleon Testbed}, author={Kate Keahey and Jason Anderson and Zhuo Zhen and Pierre Riteau and Paul Ruth and Dan Stanzione and Mert Cevik and Jacob Colleran and Haryadi S. Gunawi and Cody Hammock and Joe Mambretti and Alexander Barnes and Fran\c{c}ois Halbach and Alex Rocha and Joe Stubbs}, booktitle={Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC '20)}, publisher={USENIX Association}, month={July}, year={2020} } This information comes from the FAQs site: https://chameleoncloud.org/frequently-asked-questions/",
            "I don't know how to attach a volume to my running instance based on the provided context. The context only describes creating a new instance and does not mention attaching volumes. However, it mentions that advanced topics include 'volumes to python-chi' and 'new server methods for attaching the volume to an instance', but these are not elaborated upon in the provided text. However, I can suggest checking out the link below which is mentioned in the context as a resource for more information on this topic: https://chameleoncloud.org/docs/advanced-topics/volumes-to-python-chi This may provide more information on how to attach volumes to running instances.",
            "If you need a lease longer than 7 days, you can contact Chameleon staff via the ticketing system to request an exception. The request must be submitted by the project PI and include a detailed justification for why a contiguous lease is needed. This information comes from the FAQs site and here is the link to the site: https://docs.chameleoncloud.org/en/latest/faq.html#what-if-i-need-a-lease-that-is-longer-than-the-limitation-ie-7-days",
            "You cannot use your Chameleon allocation to mine cryptocurrency. According to the documentation, 'Resources are allocated and charged in Service Units (SUs) which equate to one hour of wall clock time on a base bare metal server.' This suggests that allocations are intended for legitimate research purposes, not for cryptocurrency mining. Source: https://chameleoncloud.org/docs/allocation-and-charging/",
            "To set up an experiment that requires a specific kernel version on a bare metal node, you can follow these steps: Create a custom image with the desired kernel version using CC-Images tool or OpenStack Diskimage-builder. Use the openstack command to upload the custom image to the target site. Reserve a bare metal node and launch an instance on it. Configure the instance to use the custom image. This information comes from the 'Bring Your Own Images' section of the Chameleon documentation, which can be found here: https://docs.chameleoncloud.org/en/latest/user/bring_your_own_images.html",
            "The Trovi artifacts repository is a service provided by Chameleon to share and access artifacts from other users on the testbed. It is integrated with the Jupyter Interface, allowing users to launch Trovi artifacts directly onto the interface and start using them. This information comes from the Chameleon documentation and can be found here: https://docs.chameleoncloud.org/en/latest/trovi.html",
            "To launch multiple nodes with one command using Heat templates, you can use the OpenStack Heat orchestration service. According to the context, Complex Appliances in Chameleon are currently implemented using OpenStack Heat. You can create a YAML file using the Heat Orchestration Template (HOT) that defines your deployment and configuration. This template will specify the number of nodes, roles, and scripts to be executed on boot. Here is an example from the provided context: openstack --os-region-name <source_site [CHI@TACC or CHI@UC]> image save <image_name> --file <filename> This command saves an image to a local file. You can then use this image and template combination (Complex Appliances) to deploy your cluster with one click. Source: This information comes from the Complex Appliances guide on Chameleon's documentation site, specifically the section on using OpenStack Heat orchestration service. Here is the link to the site: https://docs.chameleoncloud.org/en/latest/complex_appliances.html",
            "The main Chameleon hardware sites are located at: Texas Advanced Computing Center (TACC) in Austin, TX - CHI@TACC University of Chicago (UC) in Chicago, IL - CHI@UC This information comes from the Getting Started guide and here is the link to the site: https://docs.chameleoncloud.org/en/latest/getting-started.html",
            "The instance can't access the internet even with a floating IP because the security group may not allow incoming SSH (TCP port 22) traffic by default. To enable it, refer to our documentation on security group. This information comes from the FAQs site and here is the link to the site: https://chameleoncloud.org/support/faqs/",
            "To repeat a networking experiment on Chameleon, you can follow these steps: Find the experiment pattern in our tutorials on Trovi. Replicate the experiment using a different Node Type to ensure reproducibility. This information comes from the 'Welcome to Chameleon' page and here is the link to the site: https://chameleoncloud.org/welcome-to-chameleon/"]









if __name__ == '__main__':
    evaluate_score(ground_truth, rag_answer)





