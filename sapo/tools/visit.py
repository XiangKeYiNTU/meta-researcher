# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Optional
from uuid import uuid4
import json

# from verl.utils.reward_score import gsm8k
from sapo.tools.utils import visit, summarize_website
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VisitTool(BaseTool):
    """A tool for visiting URL links.

    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "visit",
                "description": "A tool for searching the web using serpapi",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "url to visit",
                        },
                        "topic": {
                            "type": "string",
                            "description": "topic for extracting information on the website",
                        }
                    },
                    "required": ["url"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Load tool config
        self.jina_api_key = config.get("jina_api_key")
        assert self.jina_api_key, "Configuration must include 'jina_api_key'"
        self.openrouter_api_key = config.get("openrouter_api_key")
        assert self.openrouter_api_key, "Configuration must include 'openrouter_api_key'"
        self.summarize_model = config.get("summarize_model", "deepseek/deepseek-chat-v3-0324:free")
        # assert self.summarize_model, "Configuration must include 'summarize_model'"

        if self.jina_api_key == "":
            raise ValueError("jina_api_key is not set")
        if self.openrouter_api_key == "":
            raise ValueError("openrouter_api_key is not set")
        # if self.summarize_model == "":
        #     raise ValueError("api_key is not set")

        logger.info(f"Initialized VisitTool with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "reward": [],
        }
        return instance_id, ToolResponse()
    
    # def execute_search(self, instance_id: str, query: str):
    #     """Execute search operation using retrieval service.

    #     Args:
    #         instance_id: Tool instance ID
    #         query: Search query
    #     Returns:
    #         A list of parsed search results
    #     """

    #     search_results = get_text_search_results(query=query, api_key=self.api_key)
    #     logger.debug(f"Search result for instance {instance_id}: {search_results}")

    #     return search_results

    def execute_visit(self, instance_id: str, url: str, topic: str = None):
        raw_content = visit(url=url, api_key=self.jina_api_key)

        content = summarize_website(content=raw_content, api_key=self.openrouter_api_key, model=self.summarize_model, topic=topic)
        logger.debug(f"Visit result for instance {instance_id}: {content}")

        return content



    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing url and topic (optional)

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        url = parameters.get("url", "")
        if not isinstance(url, str):
            url = str(url)
        if url == "":
            return ToolResponse(text="No URLs provided"), 0.0, {}
        topic = parameters.get("topic", None)

        try:
            web_content = self.execute_visit(instance_id=instance_id, url=url, topic=topic)
            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(web_content.strip())

            return ToolResponse(text=web_content), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]