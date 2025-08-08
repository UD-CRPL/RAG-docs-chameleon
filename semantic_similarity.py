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

rag_answer=["The purpose of Chameleon Associate Sites is to allow providers to add resources to the Chameleon testbed while retaining their own project identity. This allows users to discover and use these resources while still being managed by the provider's team. This information comes from the context provided and can be found in the section 'Chameleon Associate' under the 'We have identified demand for three types of scenarios...' paragraph. Source: https://docs.chameleoncloud.org/en/latest/user/associate_sites.html",
            "The specific model of GPU available on the gpu_rtx_8000 nodes is Nvidia RTX 6000. This information comes from the Chameleon Documentation and can be found here: https://docs.chameleoncloud.org/en/latest/user/gpu-support-status.html.",
            "To create an isolated network for my instances using the GUI, I need to go to the 'Advanced Networking' section and select 'Isolated networks'. This is because isolated networks allow me to create private Layer-2 VLANs for multi-node experiments.Source: Chameleon Cloud documentation, link: https://docs.chameleoncloud.org/en/latest/user/advanced-networking.html#isolated-networks",
            "To find and use a pre-configured Jupyter notebook on Chameleon, follow these steps: Go to the Chameleon home page and click on the 'Experiment' tab. Select the 'Jupyter Interface' item from the dropdown menu. This will launch a new window which will begin loading the Jupyter server. This information comes from the Jupyter Interface documentation on the Chameleon Cloud Documentation site: https://docs.chameleoncloud.org/en/latest/user/jupyter_interface.html",
            "The issue you're experiencing is not related to the context provided. The context discusses issues with launching instances and reservations, but it doesn't mention nodes stuck in the 'deploying' state. However, I can suggest that you try checking the status of your node and lease as described in the documentation on making reservations. If none of these solve your problem, please open a ticket with our help desk. This information comes from the FAQs site: https://docs.chameleoncloud.org/en/latest/faq.html",
            "Here are the answers to your questions:Is there a way to figure out if the password entered is correct? No, there isn't a way to verify passwords. If you think you have forgotten your password, we can reset it via the help desk.Source: Openstack python client issue: The request you have made requires authentication. Do I have the same problem? It seems that others have experienced similar issues, but it's not clear if it's a widespread problem. Mark mentions an intermittent issue with KVM upgrades, which was fixed. Source: Openstack python client issue: The request you have made requires authentication",
            "CHI-in-a-box is a packaging of the implementation of the core services that together constitute the Chameleon testbed for experimental Computer Science research. It allows users to discover, allocate, configure, and monitor resources on the Chameleon infrastructure. This information comes from the FAQs site: https://www.chameleoncloud.org/helpdesk/",
            "According to the context, 'deep reconfigurability'is not explicitly mentioned in the provided text. However, it can be inferred that Chameleon supports deep reconfigurability due to its ability to provide access to bare-metal machines with optional GPU and interconnection choices, as well as support for FPGAs (Field-Programmable Gate Arrays) which allow users to customize their hardware configuration.Source: Experiment Implementation on Chameleon and FPGAs — Chameleon Cloud Documentation",
            "A Floating IP is a publicly accessible IP address that can be associated with an instance, allowing it to be accessed over the public Internet. A Private IP, on the other hand, is not publicly accessible and is used for internal communication within the Chameleon Cloud. This information comes from the Chameleon Cloud documentation, specifically the section on 'Step 1: Associate an IP' and 'Step 2: Accessing Your Instance'.",
            "You should use an FPGA node when you need to perform tasks that require the unique capabilities of Field-Programmable Gate Arrays, such as high-speed data processing and acceleration. On the other hand, a GPU node is more suitable for general-purpose computing and deep learning workloads.This information comes from the Chameleon documentation on FPGA nodes and their specifications: https://chameleoncloud.org/docs/fpga/",
            "To cite Chameleon in your research paper, you should use the following reference:Keahey, K., Anderson, J., Zhen, Z., Riteau, P., Ruth, P., Stanzione, D., ... & Stubbs, J. (2020). Lessons Learned from the Chameleon Testbed. In Proceedings of the 2020 USENIX Annual Technical Conference (USENIX ATC '20) (pp. July 2020). You can find this information in the provided context under the section 'How should I cite Chameleon?'. Source: https://chameleoncloud.org/papers.html",
            "To attach a volume to your running instance, follow the steps in the 'Working with Notebooks' section. Specifically, you can use the new server methods for attaching the volume to an instance, which are now supported by python-chi. This information comes from the Tips and Tricks category on the Chameleon site: https://docs.chameleoncloud.org/tips-and-tricks.html",
            "The maximum duration for a single lease reservation is 7 days. This information comes from the context provided under 'General' section where it states: 'To create an on-demand lease, we can click next, because the form will auto-populate with defaults for the duration of the lease (the default is a one-day lease that starts immediately). If you want to change the duration of your lease or to make an advanced reservation, you can input the start date and time, number of days (maximum 7 days), and the end time. 'Source: https://docs.chameleoncloud.org/en/latest/user/reservations.html",
            "You cannot use your Chameleon allocation to mine cryptocurrency. This information comes from the context provided under 'Note: SU Costs for standard flavors are calculated based on a 48-core host.' There is no specific mention of cryptocurrency mining in the provided documentation. Source: Experiment Implementation on Chameleon and Note section",
            "To set up an experiment that requires a specific kernel version on a bare metal node, you can use the Bare Metal Experiment Pattern from the Trovi repository. Once launched, you can upload a script to the instance via SSH and run it using the run command. However, I don't know how to specify a specific kernel version during this process. This information comes from the quickstart guide on the Chameleon website: https://chameleoncloud.org/tutorials/quickstart-guide/.",
            "The Trovi artifacts repository is a collection of publicly available digital artifacts that can be used for experimentation and education. It is integrated with the Jupyter Interface, allowing users to launch artifacts directly onto the interface or upload their own artifacts from Jupyter. The repository can be accessed by going to the 'Experiment' tab on the Chameleon home page and clicking on the 'Trovi' menu item. This information comes from the context provided and here is the link to the site: https://docs.chameleoncloud.org/en/latest/user/trovi.html",
            "To launch multiple nodes with one command using Heat templates, you need to create a Complex Appliance which includes the image, template, and other configuration details. You can then use the Chameleon Command Line Interface (CLI) to deploy the Complex Appliance. This information comes from the 'Complex deployments' section under 'Advanced Features' in the provided context: https://docs.chameleoncloud.org/en/latest/quickstart.html#advanced-features",
            "The main Chameleon hardware sites located are: Texas Advanced Computing Center (TACC): Austin, TX - CHI@TACC University of Chicago (UC): Chicago, IL - CHI@UC This information comes from the 'Warm Up: Pick Your Hardware' section and here is the link to the site: https://docs.chameleoncloud.org/en/latest/warm_up/pick_your_hardware.html",
            "Your instance may not be able to access the internet even with a floating IP if it has not finished launching yet. It's recommended to wait until your instance is running before associating a floating IP address. This information comes from the Chameleon Cloud documentation: https://docs.chameleoncloud.org/en/latest/user/associate_ip.html",
            "To repeat a networking experiment found on the Chameleon blog, follow these steps: Go to the Trovi sharing portal to package and share the complete experimental environment. Find the shared environment that matches your desired experiment.Replicate the experiment by launching the shared environment on Chameleon. This information comes from the Experiment Implementation on Chameleon page and the Trovi sharing portal section, which can be found here: https://docs.chameleoncloud.org/en/latest/Experiment_Implementation.html#collaboration-reproducibility"]


if __name__ == '__main__':

    for i, (ground_truth, rag_answer) in enumerate(zip(ground_truth, rag_answer),1): 
        ground_truth = cleaning_txt(ground_truth)
        rag_answer = cleaning_txt(rag_answer)

        vector = CountVectorizer().fit_transform([ground_truth, rag_answer]).toarray()
        score = cosine_similarity_func(vector[0], vector[1])
        print(f"test{i}: {score:.4f}")

