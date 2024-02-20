from typing import Optional, Union
from graphviz import Digraph

from anytree import Node, RenderTree
import plotly.graph_objects as go
from sympy import symbols
from abc import ABC, abstractmethod
from inspect import signature
from dataclasses import dataclass, asdict


class SubroutineModel(ABC):
    def __init__(self, task_name: str, requirements: Optional[dict] = None, **kwargs):
        self.task_name = task_name
        self.requirements = requirements or {}
        self.number_of_times_called: Optional[Union[float, int]] = None
        for attr, value in kwargs.items():
            if isinstance(value, SubroutineModel):
                setattr(self, attr, value)
       
    def set_requirements(self,requirements:dataclass):
        if isinstance(requirements, dataclass):
            self.requirements = asdict(requirements)
            if "failure_tolerance"  not in self.requirements:
                RuntimeError("Failure Tolerance is a necessary requirement for resource estimation")
    
    @abstractmethod
    def populate_requirements_for_subroutines(self):
        pass

    def count_qubits(self):
        # For a generic SubroutineModel object, a symbol with the task name is returned
        # for the number of qubits
        return symbols(f"{self.task_name}_qubits")

    def run_profile(self, verbose=False):
        # Recursively populate requirements for subroutines

        if verbose:
            # If verbose, print the task name and requirements before populating.
            print(
                "Running profile for",
                self.__class__.__name__,
                f"({self.task_name})",
            )
        self.populate_requirements_for_subroutines()
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.run_profile(verbose=verbose)
                if verbose:
                    # If verbose, indicate that branch of tree has ended.
                    print("Branch ended")

    def print_profile(self, level=0):
        # Format the requirements string to be more readable
        requirements_str = "\n".join(
            "  " * (level + 2) + f"{k}: {v}" for k, v in self.requirements.items()
        )
        print_str = (
            f"{'  ' * level}Subroutine: {type(self).__name__} (Task: {self.task_name})"
        )
        print(print_str)
        if (
            requirements_str.strip()
        ):  # only print "Requirements:" if there are requirements
            print("  " * (level + 1) + "Requirements:")
            print(requirements_str)
        if self.number_of_times_called is not None:
            print("  " * (level + 1) + f"Count: {self.number_of_times_called}")
        # Recurse for child subroutines
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.print_profile(level + 1)

    def count_subroutines(self):
        counts = {self.task_name: 1}
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child_counts = child.count_subroutines()
                for name, child_count in child_counts.items():
                    if name in counts:
                        counts[name] += child_count * (
                            child.number_of_times_called
                            if child.number_of_times_called is not None
                            else 1
                        )
                    else:
                        counts[name] = child_count * (
                            child.number_of_times_called
                            if child.number_of_times_called is not None
                            else 1
                        )
        return counts

    def print_qubit_usage(self, level=0):
        # Print the number of qubits for the current subroutine
        qubits = self.count_qubits()
        print(f"{'  ' * level}Subroutine: {self.task_name} - Qubits: {qubits}")

        # Recursively call this method for all child subroutines
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.print_qubit_usage(level + 1)

    def display_hierarchy(self, graph=None, parent_name=None):
        if graph is None:
            graph = Digraph(comment="Subroutines Hierarchy")

        # Creating a node for the current subroutine
        node_name = f"{type(self).__name__}_{self.task_name}"
        graph.node(node_name, f"{type(self).__name__}\n(Task: {self.task_name})")

        # Creating an edge from the parent node to the current node
        if parent_name:
            graph.edge(parent_name, node_name)

        # Recurse for child subroutines
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                child.display_hierarchy(graph, node_name)

        return graph

    def create_tree(self, parent=None):
        class_name = self.__class__.__name__
        # Format the task name with the class name in parentheses.
        display_name = f"{self.task_name} ({class_name})"

        if parent is None:
            self.node = Node(display_name)
        else:
            self.node = Node(display_name, parent=parent.node)

        for attr in dir(self):
            child = getattr(self, attr)
            # Ensure the attribute is an instance of SubroutineModel or its subclasses.
            if isinstance(child, SubroutineModel) and attr != "node":
                child.create_tree(self)

    def display_tree(self, parent=None):
        self.create_tree(parent=parent)
        for pre, _, node in RenderTree(self.node):
            treestr = "%s%s" % (pre, node.name)
            print(treestr.ljust(8))

    def build_graph_data(
        self, edges, nodes, parent_name=None, depth=0, sibling_index=0
    ):
        node_name = f"{self.task_name}"
        if node_name not in nodes:
            # Set the horizontal position based on sibling index and vertical position based on depth
            nodes[node_name] = {
                "requirements": self.requirements,
                "pos": (
                    sibling_index * 10,
                    -depth * 5,
                ),  # Spacing can be adjusted as needed
            }

        if parent_name is not None:
            edges.append((parent_name, node_name))

        # Initialize sibling index for child nodes
        child_sibling_index = 0
        for attr in dir(self):
            child = getattr(self, attr)
            if isinstance(child, SubroutineModel):
                # Pass the sibling index to space out nodes at the same depth
                child.build_graph_data(
                    edges, nodes, node_name, depth + 1, child_sibling_index
                )
                child_sibling_index += (
                    1  # Increment the sibling index for the next sibling
                )

    def plot_graph(self):
        edges = []
        nodes = {}
        self.build_graph_data(edges, nodes)

        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = nodes[edge[0]]["pos"]
            x1, y1 = nodes[edge[1]]["pos"]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_x = []
        node_y = []
        hover_text = []
        for node in nodes:
            x, y = nodes[node]["pos"]
            node_x.append(x)
            node_y.append(y)
            requirements_str = "<br>".join(
                [f"{k}: {v}" for k, v in nodes[node]["requirements"].items()]
            )
            hover_text.append(requirements_str)

        # Use text for nodes and style it to appear as if it's inside ovals
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="text",
            text=[node for node in nodes],
            hoverinfo="text",
            hovertext=hover_text,
            textfont=dict(size=10, color="white"),
            textposition="middle center",
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Subroutine Hierarchy",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
                annotations=[
                    go.layout.Annotation(
                        x=nodes[node]["pos"][0],
                        y=nodes[node]["pos"][1],
                        text=node,  # Node labels (task names)
                        showarrow=False,
                        font=dict(size=12, color="black"),
                        width=8 * len(node),  # Approx width of text
                        height=20,
                        bgcolor="lightblue",  # Background color
                        bordercolor="blue",
                        borderpad=4,  # Padding around the text
                        borderwidth=1,
                    )
                    for node in nodes
                ],
            ),
        )

        fig.show()
