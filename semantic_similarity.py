from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re

#The sentences are identical so the score should be 1.0
#ground_truth = "I like the blue sky"
#rag_answer = "I like the blue sky"

#The sentences share some words, so the score should be between 0.0 and 1.0
#ground_truth = "I like machine learning"
#rag_answer = "Sky is blue"

def cleaning_txt(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

#compute cosine similarity function
def cosine_similarity_func(x,y):
    #Ensure two vectors have the same length
    if len(x) != len(y):
        return None

    dot_product = np.dot(x,y)

    #compute magnitudes of x & y
    magnitude_x = np.sqrt(np.sum(x**2))
    magnitude_y = np.sqrt(np.sum(y**2))

    cosine_similarity = dot_product / (magnitude_x * magnitude_y)

    return cosine_similarity 


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

rag_answer=["The purpose of the Chameleon Associate Sites is to allow providers to add resources to the Chameleon testbed while retaining their own project identity, and provide system administration of their resources with support from the Chameleon team. This type of provider will also use Chameleon user services, such as user/project management, user portal, resource discovery, and appliance catalog. This information comes from the context provided, specifically from the section 'Chameleon Associate: In this scenario a provider wants to add resources to the Chameleon testbed...'. Source: https://chameleoncloud.org/faq (Frequently Asked Questions)",
        "According to the provided context, the specific model of GPU available on the gpu_rtx_8000 nodes is not explicitly mentioned. However, it does mention that 'Nvidia RTX 6000' is supported. This information comes from the Chameleon documentation and can be found here: link to the page.",
        "I don't know how to create an isolated network for my instances using the GUI from the provided context. The context only mentions creating private Layer-2 VLANs for multi-node experiments under 'Advanced Networking' but does not provide instructions on how to do it through the GUI. However, I can suggest checking the Chameleon Cloud Documentation's FAQs or Getting Started guide for more information on this topic.",
        "To find and use a pre-configured Jupyter notebook on Chameleon, follow these steps: Go to the Chameleon home page and click on the 'Experiment' tab. Select the 'Jupyter Interface' item from the dropdown menu. This will launch a new window which will begin loading the Jupyter server. This information comes from the Getting Started guide and can be found here: https://docs.chameleoncloud.org/en/latest/getting_started.html#jupyter-on-chameleon",
        "I don't know the specific solution for a bare metal node stuck in the 'deploying' state. However, I can suggest checking the documentation on troubleshooting or contacting the Help Desk for assistance. This information comes from the FAQs site: https://docs.chameleoncloud.org/en/latest/getting-started.html",
        "The question is about an 'Error 403: Forbidden' when trying to use the OpenStack CLI. According to the context, the issue might be due to authentication problems. The user has sourced the CHI-openrc.sh file and entered their password, but still receives a 'The request you have made requires authentication.' error message. This information comes from the FAQs site: https://support.chameleoncloud.org/hc/en-us/articles/360037111333-Openstack-python-client-issue-The-request-you-have-made-requires-authentication-General-User-Questions To troubleshoot this issue, I would recommend checking the OpenStack documentation for troubleshooting authentication issues.",
        "CHI-in-a-box is a packaging of the implementation of the core services that together constitute the Chameleon testbed for experimental Computer Science research. It allows users to discover information, allocate resources, configure them, and monitor various types of metrics. This information comes from the CHI-in-a-box documentation, which can be found here: https://docs.chameleoncloud.org/en/latest/user/chi_in_a_box.html.",
        "According to the context, 'deep reconfigurability' refers to the ability to fully control over physical hardware and software stack, enabling reproducible experimental research. This is mentioned in the first paragraph of the documentation: 'Chameleon provides researchers with deeply reconfigurable cloud infrastructure for systems, networking, distributed computing, and security research.' Source: Chameleon Cloud Documentation - https://chameleoncloud.org/docs/",
        "A Floating IP is a publicly accessible IP address associated with an instance, allowing external connections to the instance. On the other hand, a Private IP is not publicly accessible and is used for communication between instances within the same project. This information comes from the Quickstart Guide and can be found here: https://docs.chameleoncloud.org/en/latest/quickstart.html",
        "You should use an FPGA node when: You need a custom hardware acceleration for specific tasks such as AI, machine learning, or data processing. You require low latency and high throughput for applications like networking, security, or scientific simulations. This information comes from the Chameleon documentation and here is the link to the site: https://chameleoncloud.org/ Note that FPGA nodes are suitable for custom hardware acceleration and specific tasks that require low latency and high throughput.",
        "To cite Chameleon in your research paper, you should use the following reference Keahey, K., Anderson, J., Zhen, Z., Riteau, P., Ruth, P., Stanzione, D., ... & Stubbs, J. (2020). Lessons Learned from the Chameleon Testbed. In Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC '20) (pp. 1-12). USENIX Association. This information comes from the FAQs site and here is the link to the site: https://chameleoncloud.org/docs/faqs.html",
        "I don't have information on how to attach a volume to a running instance in the provided context. However, I can suggest checking the documentation for more information. This information comes from the FAQs site: https://chameleoncloud.org/docs/.",
        "The maximum duration for a single lease reservation is 7 days, as stated in the 'Plan your workflow' section: 'Remember that leases last up to one week...'. This information comes from the FAQs site and here is the link to the site: https://chameleoncloud.org/docs/quickstart-guide.",
        "You cannot use your Chameleon allocation to mine cryptocurrency. According to the context, Chameleon's primary purpose is for research and experimentation, and it has a 'soft allocation model' where each project receives a startup allocation of 20,000 SUs for six months. This information comes from the FAQs site and here is the link to the site: https://chameleoncloud.org/docs/faq.",
        "To set up an experiment that requires a specific kernel version on a bare metal node, you can use the ssh module in Chameleon's API. Specifically, you can use the Remote class to connect to your instance and run a script that sets up the kernel version. Here is an example code snippet:from chi import ssh with ssh.Remote(floating_ip) as conn:# Upload the script conn.put('setup.sh') # Run the script conn.run('bash setup.sh') This information comes from the 'Experiment Implementation on Chameleon' section of the documentation, which can be found here: https://docs.chameleoncloud.org/en/latest/user/quickstart.html#experiment-implementation-on-chameleon Note that this assumes you have already created a lease and launched an instance on the bare metal node.",
        "The Trovi artifacts repository is a collection of experiment patterns and artifacts that can be used on the Chameleon testbed. It allows users to launch and view Trovi artifacts, including those from other users, directly onto the Jupyter Interface. This information comes from the Experiment Implementation on Chameleon section and here is the link to the site: https://docs.chameleoncloud.org/en/latest/user/experiment_implementation.html",
        "To launch multiple nodes with one command using Heat templates, you need to create a Complex Appliance document (template) that contains the information needed to orchestrate the deployment and configuration of your cluster. This template will specify not only what image you want to deploy but also on how many nodes you want to deploy that image. This information comes from the Experiment Implementation on Chameleon section, specifically the part about Complex Appliances. Here is a link to the relevant documentation: https://docs.chameleoncloud.org/en/latest/user/complex_appliances.html",
        "The main Chameleon hardware sites located are:CHI@TACC (Texas Advanced Computing Center): https://chi.tacc.chameleoncloud.org CHI@UC (University of Chicago): https://chi.uc.chameleoncloud.org CHI@NCAR (National Center for Atmospheric Research): https://chi.ncar.chameleoncloud.org CHI@Edge (Edge computing testbed): https://chi.edge.chameleoncloud.org This information comes from the 'Step 1: Access a Testbed Site' section of the Chameleon documentation.",
        "The instance cannot access the internet even with a floating IP because it needs to be running before associating the floating IP. This is stated in the 'Getting Started' section of the documentation: 'best to wait until your instance is running before doing this step to ensure no issues.' Source: Chameleon Cloud Documentation, Getting Started section (link:https://docs.chameleoncloud.org/en/latest/getting-started.html)",
        "To repeat a networking experiment found on the Chameleon blog, follow these steps: Visit the Trovi sharing portal to package and share the complete experimental environment. Use the packaged environment to recreate the experiment. This information comes from the 'Collaboration & Reproducibility' section of the Chameleon documentation. Source: https://chameleoncloud.org/docs/collaboration-reproducibility/"]


if __name__ == '__main__':

    for i, (ground_truth, rag_answer) in enumerate(zip(ground_truth, rag_answer),1): 
        ground_truth = cleaning_txt(ground_truth)
        rag_answer = cleaning_txt(rag_answer)

        vector = CountVectorizer().fit_transform([ground_truth, rag_answer]).toarray()
        score = cosine_similarity_func(vector[0], vector[1])
        print(f"test{i}: {score:.4f}")

