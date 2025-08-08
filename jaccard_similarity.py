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


rag_answer=[]










if __name__ == '__main__':
    evaluate_score(ground_truth, rag_answer)





