import json
from typing import List, Optional

from agno.document import Document
from agno.knowledge.website import WebsiteKnowledgeBase
from agno.knowledge.combined import CombinedKnowledgeBase  # 新增导入
from agno.tools import Toolkit
from agno.utils.log import logger


class WebsiteTools(Toolkit):
    def __init__(
        self,
        knowledge_base: Optional[WebsiteKnowledgeBase] = None,
        combined_knowledge_base: Optional[CombinedKnowledgeBase] = None,  # 新增参数
    ):
        super().__init__(name="website_tools")
        self.knowledge_base: Optional[WebsiteKnowledgeBase] = knowledge_base
        self.combined_knowledge_base: Optional[CombinedKnowledgeBase] = combined_knowledge_base  # 新增参数

        if self.knowledge_base is not None and isinstance(self.knowledge_base, WebsiteKnowledgeBase):
            self.register(self.add_website_to_knowledge_base)
        else:
            self.register(self.read_url)

    def add_website_to_knowledge_base(self, url: str) -> str:
        """This function adds a website's content to the knowledge base.
        NOTE: The website must start with https:// and should be a valid website.

        USE THIS FUNCTION TO GET INFORMATION ABOUT PRODUCTS FROM THE INTERNET.

        :param url: The URL of the website to add.
        :return: 'Success' if the website was added to the knowledge base.
        """
        if self.knowledge_base is None:
            return "Knowledge base not provided"

        logger.debug(f"Adding to knowledge base: {url}")
        self.knowledge_base.urls.append(url)

        logger.debug("Loading website knowledge base.")
        self.knowledge_base.load(recreate=False)

        # 如果配置了 CombinedKnowledgeBase，则在 website_knowledge_base 更新后也更新它
        if self.combined_knowledge_base is not None:
            logger.debug("Updating combined knowledge base after website update.")
            self.combined_knowledge_base.load(recreate=True)  # 强制重载，确保数据同步

        return "Success"

    def read_url(self, url: str) -> str:
        """This function reads a URL and returns the content.

        :param url: The URL of the website to read.
        :return: Relevant documents from the website.
        """
        from agno.document.reader.website_reader import WebsiteReader

        website = WebsiteReader()

        logger.debug(f"Reading website: {url}")
        relevant_docs: List[Document] = website.read(url=url)
        return json.dumps([doc.to_dict() for doc in relevant_docs])
