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
from sapo.tools.utils import get_text_search_results
from verl.utils.rollout_trace import rollout_trace_op

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchTool(BaseTool):
    """A tool for conducting web searches using serpapi.

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
                "name": "search",
                "description": "A tool for searching the web using serpapi",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query for searching",
                        },
                    },
                    "required": ["query"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Load tool config
        self.api_key = config.get("api_key")
        assert self.api_key, "Configuration must include 'api_key'"

        if self.api_key == "":
            raise ValueError("api_key is not set")

        logger.info(f"Initialized SearchTool with config: {config}")

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
    
    def execute_search(self, instance_id: str, query: str):
        """Execute search operation using retrieval service.

        Args:
            instance_id: Tool instance ID
            query: Search query
        Returns:
            A list of parsed search results
        """

        search_results = get_text_search_results(query=query, api_key=self.api_key)
        logger.debug(f"Search result for instance {instance_id}: {search_results}")

        return search_results



    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute the search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing query

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query = parameters.get("query", "")
        if not isinstance(query, str):
            query = str(query)

        if query == "":
            return ToolResponse(text="No search query provided"), 0,0, {}

        try:
            search_results = self.execute_search(instance_id=instance_id, query=query)
            result_text = json.dumps(search_results, indent=4)
            # Store results in instance dictionary
            self._instance_dict[instance_id]["reward"].append(result_text.strip())

            return ToolResponse(text=result_text), 0.0, {}
        except Exception as e:
            return ToolResponse(text=f"Error: {e}"), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> str:
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]