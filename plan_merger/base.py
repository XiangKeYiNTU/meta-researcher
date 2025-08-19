from collections import deque

from tree_search.schemas import Plan, Step

class StepNode:
    def __init__(self, step: Step):
        self.step = step
        self.children = []
        # self.parent = None
        self.execution_result = None
        

class PlanGraph:
    def __init__(self):
        self.start_node = StepNode(step=Step(goal="START", instructions="START"))
        self.start_node.execution_result = "Started the execution"
        self.end_node = StepNode(step=Step(goal="END", instructions="END"))
        self.node_list = [self.start_node, self.end_node]

    def exist_step(self, step: Step):
        for node in self.node_list:
            if node.step == step:
                return node
        return None
    

    def add_plan(self, plan: Plan):
        for i in range(len(plan.steps) - 1):
            cur_step = plan.steps[i]
            next_step = plan.steps[i + 1]

            # Get or create current node
            cur_node = self.exist_step(cur_step)
            if not cur_node:
                cur_node = StepNode(step=cur_step)
                self.node_list.append(cur_node)
                if i == 0:
                    self.start_node.children.append(cur_node)


            # Get or create next node
            next_node = self.exist_step(next_step)
            if not next_node:
                next_node = StepNode(step=next_step)
                self.node_list.append(next_node)
                if i == (len(plan.steps) - 2):
                    next_node.children.append(self.end_node)

            # Add next_node as a child if not already linked
            if next_node not in cur_node.children:
                cur_node.children.append(next_node)

    def add_plan_list(self, plan_list: list[Plan]):
        for plan in plan_list:
            self.add_plan(plan)

    def get_next_exec_steps(self):
        visited = set()
        queue = deque([self.start_node])
        next_exec_nodes = []

        while queue:
            node = queue.popleft()

            if node in visited:
                continue
            visited.add(node)

            # Skip START and END nodes - they are not executable steps
            if node.step.goal in ["START"]:
                # Add children to continue traversal
                for child in node.children:
                    if child not in visited:
                        queue.append(child)
                continue

            # If execution_result is None, it's the next step to execute
            if node.execution_result is None:
                next_exec_nodes.append(node)
                # Do not continue exploring this branch, wait for this node to be executed
                continue

            # Otherwise, add its children to explore further
            for child in node.children:
                if child not in visited:
                    queue.append(child)

        return next_exec_nodes
    
    def get_current_exec_results(self):
        """
        Get all the current execution results.
        Returns a list of tuples (Step, execution_result).
        """
        results = []

        # Traverse all nodes in the plan graph
        for node in self.node_list:
            # Add step and its execution result if available (excluding START/END)
            if node.execution_result is not None and node.step.goal not in ["START", "END"]:
                results.append((node.step, node.execution_result))

        return results
