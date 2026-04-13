import os
import requests
from urllib.parse import urljoin
from langchain_core.documents import Document
from bs4 import BeautifulSoup

os.environ['USER_AGENT'] = 'myagent'
headers = {"User-Agent": os.environ['USER_AGENT']}

READTHEDOCS_BASE = "https://chameleoncloud.readthedocs.io/en/latest/"
PYTHON_CHI_BASE = "https://python-chi.readthedocs.io/en/latest/"
CHI_EDGE_GITBOOK = "https://chameleoncloud.gitbook.io/chi-edge"
TROVI_GITBOOK = "https://chameleoncloud.gitbook.io/trovi"

# chameleoncloud.org pages worth indexing (filtered from sitemap)
CHAMELEON_ORG_URLS = [
    "https://chameleoncloud.org/learn/frequently-asked-questions/",
    "https://chameleoncloud.org/about/chameleon/",
    "https://chameleoncloud.org/experiment/sites/",
    "https://chameleoncloud.org/hardware/",
    "https://chameleoncloud.org/experiment/chiedge/",
    "https://chameleoncloud.org/experiment/chiedge/hardware-info/",
]

# Blog posts (no sitemap — manually curated, high-value content)
BLOG_BASE = "https://blog.chameleoncloud.org"
# Only index posts from these categories — skip user experiments and announcements
# which tend to have little practical how-to content
BLOG_CATEGORIES = {"tips-and-tricks", "chameleon-changelog", "featured"}

# Forum posts (high-value Q&A)
FORUM_BASE = "https://forum.chameleoncloud.org"


def get_readthedocs_urls(base_url):
    """Discover all page URLs from the readthedocs table of contents page."""
    toc_url = urljoin(base_url, "contents.html")
    try:
        resp = requests.get(toc_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if href.endswith(".html") and not any(x in href for x in ["genindex", "search", "py-modindex"]):
                urls.add(urljoin(base_url, href))
        return sorted(urls)
    except Exception as e:
        print(f"Failed to fetch TOC from {toc_url}: {e}")
        return []


def get_python_chi_urls(base_url):
    """Discover all page URLs from the python-chi index page."""
    try:
        resp = requests.get(base_url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        urls = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].split("#")[0]
            if href.endswith(".html") and not any(x in href for x in ["genindex", "search", "py-modindex"]):
                full = urljoin(base_url, href)
                if full.startswith(base_url):  # only keep urls within python-chi
                    urls.add(full)
        return sorted(urls)
    except Exception as e:
        print(f"Failed to fetch python-chi index from {base_url}: {e}")
        return []


def get_gitbook_docs(base_url=CHI_EDGE_GITBOOK, skip_prefixes=()):
    """Fetch all pages from a GitBook site using its sitemap and .md endpoints."""
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    docs = []
    try:
        resp = requests.get(f"{base_url}/sitemap-pages.xml", headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        urls = [loc.text.strip() for loc in soup.find_all("loc")]
    except Exception as e:
        print(f"Failed to fetch GitBook sitemap: {e}")
        return docs

    for url in urls:
        if any(url.startswith(f"{base_url}/{p}") for p in skip_prefixes):
            continue

        # GitBook root URL maps to readme.md; all other pages use {url}.md
        md_url = f"{base_url}/readme.md" if url.rstrip("/") == base_url.rstrip("/") else f"{url}.md"
        try:
            md_resp = requests.get(md_url, headers=headers, timeout=10)
            md_resp.raise_for_status()
            if md_resp.text.strip().startswith("# Page Not Found"):
                continue
            docs.append(Document(page_content=md_resp.text,
                                 metadata={"source": url, "source_type": "gitbook"}))
        except Exception as e:
            print(f"Failed to fetch GitBook page {url}: {e}")

    return docs


def get_blog_urls(base_url=BLOG_BASE, categories=BLOG_CATEGORIES):
    """Crawl all paginated blog listing pages and collect post URLs in allowed categories."""
    post_urls = set()
    page = 1
    while True:
        listing_url = base_url if page == 1 else f"{base_url}/page/{page}/"
        try:
            resp = requests.get(listing_url, headers=headers, timeout=10)
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Collect post URLs and their categories from this page
            for article in soup.find_all("article", class_="post-item"):
                # Get categories for this post
                post_cats = {
                    a["href"].rstrip("/").split("/")[-1]
                    for a in article.find_all("a", href=True)
                    if "/categories/" in a["href"]
                }
                if not categories or post_cats & categories:
                    link = article.find("a", href=True)
                    if link and link["href"].startswith(base_url):
                        post_urls.add(link["href"].rstrip("/") + "/")

            # Check if there's a next page
            if not soup.find("a", string=lambda t: t and "next" in t.lower()):
                break
            page += 1
        except Exception as e:
            print(f"Failed to fetch blog page {page}: {e}")
            break
    return sorted(post_urls)


def get_forum_docs(base_url=FORUM_BASE):
    """Fetch all topics from the Discourse forum via JSON API and return as Documents."""
    docs = []
    try:
        resp = requests.get(f"{base_url}/latest.json", headers=headers, timeout=10)
        resp.raise_for_status()
        topics = resp.json()["topic_list"]["topics"]
    except Exception as e:
        print(f"Failed to fetch forum topic list: {e}")
        return docs

    for topic in topics:
        topic_id = topic["id"]
        title = topic["title"]
        url = f"{base_url}/t/{topic_id}"
        try:
            tresp = requests.get(f"{url}.json", headers=headers, timeout=10)
            tresp.raise_for_status()
            posts = tresp.json()["post_stream"]["posts"]

            # Concatenate all posts in the thread, labelled by position
            parts = [f"Forum topic: {title}\n"]
            for i, post in enumerate(posts):
                text = BeautifulSoup(post["cooked"], "html.parser").get_text(separator="\n")
                label = "Question" if i == 0 else f"Reply {i}"
                parts.append(f"[{label}]\n{text.strip()}")

            content = "\n\n".join(parts)
            docs.append(Document(page_content=content,
                                 metadata={"source": url, "source_type": "forum"}))
        except Exception as e:
            print(f"Failed to fetch forum topic {topic_id}: {e}")

    return docs


def _source_type(url):
    if "readthedocs.io" in url:
        return "readthedocs"
    if "python-chi" in url:
        return "python_chi"
    if "blog.chameleoncloud.org" in url:
        return "blog"
    if "forum.chameleoncloud.org" in url:
        return "forum"
    if "gitbook.io" in url:
        return "gitbook"
    if "chameleoncloud.org" in url:
        return "chameleon_org"
    return "other"


def clean_docs(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(["header", "footer", "nav", "aside", "script", "style"]):
            tag.decompose()

        # Prefer the main content area to avoid pulling in sidebars/navigation.
        # Sphinx/ReadTheDocs pages use div[role=main] or .rst-content; blog/org
        # pages typically have <main> or <article>. Fall back to the whole body.
        content_root = (
            soup.find("div", {"role": "main"})
            or soup.find("div", class_="rst-content")
            or soup.find("article")
            or soup.find("main")
            or soup
        )

        for class_name in ["toc", "toctree-wrapper", "headerlink",
                            "wy-nav-side", "rst-content-toc", "related-docs"]:
            for tag in content_root.find_all(class_=class_name):
                tag.decompose()

        raw_text = content_root.get_text(separator="\n")
        lines = [line.strip() for line in raw_text.splitlines()]
        clean_text = "\n".join(line for line in lines if line)

        return Document(page_content=clean_text,
                        metadata={"source": url, "source_type": _source_type(url)})

    except Exception as e:
        print(f"Failed to process {url}: {e}")
        return None


def loader_docs():
    readthedocs_urls = get_readthedocs_urls(READTHEDOCS_BASE)
    python_chi_urls = get_python_chi_urls(PYTHON_CHI_BASE)
    blog_urls = get_blog_urls()
    forum_docs = get_forum_docs()
    chi_edge_docs = get_gitbook_docs(CHI_EDGE_GITBOOK)
    trovi_docs = get_gitbook_docs(TROVI_GITBOOK, skip_prefixes=("api-reference", "meta"))

    all_urls = readthedocs_urls + python_chi_urls + CHAMELEON_ORG_URLS + blog_urls

    print(f"  readthedocs:   {len(readthedocs_urls)} pages")
    print(f"  python-chi:    {len(python_chi_urls)} pages")
    print(f"  chameleon.org: {len(CHAMELEON_ORG_URLS)} pages")
    print(f"  blog posts:    {len(blog_urls)} posts")
    print(f"  forum topics:  {len(forum_docs)} topics")
    print(f"  chi@edge docs: {len(chi_edge_docs)} pages")
    print(f"  trovi docs:    {len(trovi_docs)} pages")
    print(f"  total:         {len(all_urls) + len(forum_docs) + len(chi_edge_docs) + len(trovi_docs)} documents")

    docs = []
    for url in all_urls:
        doc = clean_docs(url)
        if doc:
            docs.append(doc)

    docs.extend(forum_docs)
    docs.extend(chi_edge_docs)
    docs.extend(trovi_docs)
    return docs


if __name__ == "__main__":
    docs = loader_docs()
    print(f"\nSuccessfully loaded {len(docs)} documents.")
