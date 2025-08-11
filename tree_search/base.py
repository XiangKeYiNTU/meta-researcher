import math

from tree_search.schemas import Plan, Step, PlanScore, ModificationResponse, ActionType, ActionParams
# from schemas import Plan, Step, PlanScore, ModificationResponse, ActionType, ActionParams

EXPLORATION_CONSTANT = 1.41  # Commonly used exploration constant in UCT

class BaseTreeNode:
    def __init__(self, score: PlanScore = None, parent: 'BaseTreeNode' = None, children=None, execution_flag=False, execution_result=None):
        self.score = score
        self.parent = parent
        self.children = children if children is not None else []
        self.execution_flag = execution_flag
        self.execution_result = execution_result

    def get_plan(self):
        modifications = []
        current_node = self
        while current_node.parent:
            modifications.append(ModificationResponse(
                rationale=current_node.rationale,
                action=ActionType(current_node.action),
                # action_params=ActionParams(**current_node.action_params)
                action_params=current_node.action_params
            ))
            current_node = current_node.parent
        modifications.reverse()
        return current_node.root_plan.apply_modifications(modifications)

    # def add_child(self, rationale: str, action: str, action_params: dict, score: PlanScore):
    #     child_node = ModifiedNode(rationale, action, action_params, score)
    #     self.children.append(child_node)

    def compute_uct(self):
        # prioritize unexpanded nodes
        if not self.children:
            return float('inf')
        
        parent_visits = len(self.parent.children) if self.parent else 1
        score = self.score.effectiveness + self.score.completeness + self.score.executability
        exploitation = score / len(self.children)
        exploration = EXPLORATION_CONSTANT * math.sqrt(
            math.log(parent_visits) / len(self.children)
        )
        return exploitation + exploration      

class InitialNode(BaseTreeNode):
    def __init__(self, question: str, plan: Plan, score: PlanScore = None, children=None, execution_flag=False, execution_result=None):
        super().__init__(score=score, parent=None, children=children, execution_flag=execution_flag, execution_result=execution_result)
        self.question = question
        self.root_plan = plan

class ModifiedNode(BaseTreeNode):
    def __init__(self, parent: BaseTreeNode, rationale: str, action: str, action_params: dict, score: PlanScore = None, children=None, execution_flag=False, execution_result=None):
        super().__init__(score=score, parent=parent, children=children, execution_flag=execution_flag, execution_result=execution_result)
        self.rationale = rationale
        self.action = action
        self.action_params = action_params

# class SearchTree:
#     def __init__(self, question: str, steps: list[Step], max_depth: int = 3, max_children: int = 3):
#         self.question = question
#         self.root_plan = Plan(steps=steps)
#         self.num_layers = 0
#         self.max_depth = max_depth
#         self.max_children = max_children

#     def search_for_max_score_node(self):


    # def search(self, criteria):
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def execute_action(self, node: TreeNode):
    #     raise NotImplementedError("This method should be implemented by subclasses.")

    # def evaluate_node(self, node: TreeNode):
    #     raise NotImplementedError("This method should be implemented by subclasses.")

class SearchTree:
    def __init__(self, question: str, initial_plan: Plan, initial_score: PlanScore = None, max_depth=2, max_children=2):
        self.root = InitialNode(question=question, plan=initial_plan, score=initial_score)
        self.max_depth = max_depth
        self.max_children = max_children

    def select(self):
        """
        Recursively traverse the tree and select the node with the highest UCT value for expansion.
        Only considers nodes that are not at max depth and have fewer than max_children.
        """
        best_node = None
        best_uct = float('-inf')

        def traverse(node, depth):
            nonlocal best_node, best_uct

            if depth >= self.max_depth:
                return  # Reached max depth, don't go further

            # Check if this node is expandable (not full)
            if len(node.children) < self.max_children:
                uct = node.compute_uct()
                if uct > best_uct:
                    best_uct = uct
                    best_node = node

            # Recurse on children
            for child in node.children:
                traverse(child, depth + 1)

        traverse(self.root, depth=0)
        return best_node
    
    def print_tree(self):
        """
        Traverse the tree and print each node's plan as a JSON string.
        """
        def traverse(node, depth=0):
            # Get the Pydantic plan model and convert it to an indented JSON string
            plan_model = node.get_plan()
            json_str = plan_model.model_dump_json(indent=2)  # For Pydantic v2

            # Print with indentation for structure
            print("  " * depth + f"Node at depth {depth}:")
            print("  " * depth + json_str)
            print("  " * depth + f"Score: {node.score.effectiveness}, {node.score.completeness}, {node.score.executability}")

            for child in node.children:
                traverse(child, depth + 1)

        traverse(self.root)

    def select_top_k(self, top_k : int = 3):
        """
        Select the top k nodes based on their scores.
        """
        # Select the top k nodes based on their scores
        top_k_nodes = []

        def traverse(node):
            if len(top_k_nodes) < top_k:
                top_k_nodes.append(node)
            else:
                # Find the node with the lowest score and replace it if the current node has a higher score
                min_index = min(range(len(top_k_nodes)), key=lambda i: top_k_nodes[i].score.effectiveness + top_k_nodes[i].score.completeness + top_k_nodes[i].score.executability)
                if node.score.effectiveness + node.score.completeness + node.score.executability > top_k_nodes[min_index].score.effectiveness + top_k_nodes[min_index].score.completeness + top_k_nodes[min_index].score.executability:
                    top_k_nodes[min_index] = node
            for child in node.children:
                traverse(child)

        traverse(self.root)
        
        # Select the top k plans
        top_k_plans = [node.get_plan() for node in top_k_nodes]

        return top_k_plans
        # with open(save_path, 'w') as f:
        #     f.write("[\n" + ",\n".join(serialized_plans) + "\n]")

    def select_top_plans(self):
        """
        Select all plans with the highest total score in the tree.
        Returns a list of Plan objects.
        """
        all_nodes = []

        # Step 1: Gather all nodes
        def traverse(node):
            all_nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(self.root)

        if not all_nodes:
            return []

        # Step 2: Compute scores for each node
        def total_score(node):
            return (
                node.score.effectiveness
                + node.score.completeness
                + node.score.executability
            )

        # Step 3: Find the highest score
        max_score = max(total_score(n) for n in all_nodes)

        # Step 4: Return plans with this score
        top_plans = [n.get_plan() for n in all_nodes if total_score(n) == max_score]

        return top_plans

        

if __name__ == "__main__":
    # debug
    question = "What is the capital of France?"

    step_1 = Step(
        goal="Find the capital of France",
        instructions="Search for the capital city of France using reliable sources."
    )

    step_2 = Step(
        goal="Verify the capital of France",
        instructions="Cross-check the information with multiple sources to ensure accuracy."
    )

    initial_plan = Plan(steps=[step_1, step_2])

    search_tree = SearchTree(question=question, initial_plan=initial_plan)

    # Add a modified node

    modified_node = ModifiedNode(
        parent=search_tree.root,
        rationale="The initial plan needs to be expanded to include more detailed steps.",
        action="add",
        action_params=ActionParams(
            position=1,
            goal="Check the latest news for updates on France's capital",
            instructions="Look for any recent changes or news related to the capital of France."
        )
    )

    search_tree.root.children.append(modified_node)

    # Add another modified node

    modified_node_2 = ModifiedNode(
        parent=modified_node,
        rationale="The initial plan needs to include a step for checking historical context.",
        action="remove",
        action_params=ActionParams(
            position=2,
            goal="Verify the capital of France",
            instructions="Cross-check the information with multiple sources to ensure accuracy."
        )
    )

    modified_node.children.append(modified_node_2)

    print(search_tree.root.get_plan().model_dump_json(indent=2))
    print("---- Modified Node ----")
    print(modified_node.get_plan().model_dump_json(indent=2))
    print("---- Modified Node 2 ----")
    print(modified_node_2.get_plan().model_dump_json(indent=2))

