class PassDirection:
    """
    This class defines a message passing step on the encoded query graphs, specifying how messages are passed
    between nodes based on edge types and destination node types.

    Attributes:
        etypes (set): A set of tuples representing the edge types that qualify for message passing.
        in_types (set): A set of source node types that are involved in message passing.
        out_types (set): A set of destination node types that messages are passed to.
        model_name (str): The name of the edge model used for combining messages.
    """

    def __init__(
        self,
        model_name: str,
        graph,
        e_name: str = None,
        n_dest: str = None,
        allow_empty: bool = False,
    ):
        """
        Initializes a message passing step with specified parameters.

        Args:
            model_name (str): The name of the edge model to be used for combining the messages.
            graph: The graph object on which the message passing should be performed. It should have an attribute
               `canonical_etypes` that provides the edge types in the graph.
            e_name (str, optional): The edge type to be considered for message passing. Edges are defined by triplets:
                                    (source_node_type, edge_type, destination_node_type). Only edges where edge_type
                                    matches `e_name` are included in the message passing step. Defaults to None, which
                                    means all edge types are considered.
            n_dest (str, optional): Further restricts the edges that are incorporated in the message passing by
                                    specifying the destination node type. Defaults to None, which means all destination
                                    node types are considered.
            allow_empty (bool): If True, allows the scenario where no edges in the graph qualify for this message
                                passing step. If False, raises a ValueError in such a scenario. Defaults to False.

        Raises:
            ValueError: If `allow_empty` is False and no edges in the graph qualify for the specified `e_name`
                            and `n_dest`.
        """
        self.etypes = set()
        self.in_types = set()
        self.out_types = set()
        self.model_name = model_name

        # Filter edge types based on provided e_name and n_dest, if any
        if e_name is not None and n_dest is not None:
            # Both edge and destination node types are specified for filtering
            filtered_etypes = [
                etype
                for etype in graph.canonical_etypes
                if etype[1] == e_name and etype[2] == n_dest
            ]
        elif e_name is not None:
            # Only edge type is specified for filtering
            filtered_etypes = [
                etype for etype in graph.canonical_etypes if etype[1] == e_name
            ]
        elif n_dest is not None:
            # Only destination node type is specified for filtering
            filtered_etypes = [
                etype for etype in graph.canonical_etypes if etype[2] == n_dest
            ]
        else:
            # No filtering criteria provided; use all edge types
            filtered_etypes = list(graph.canonical_etypes)

        # Process filtered edge types to populate etypes, in_types, and out_types
        for curr_n_src, curr_e_name, curr_n_dest in filtered_etypes:
            # Add the current edge type to the set of edge types
            self.etypes.add((curr_n_src, curr_e_name, curr_n_dest))
            # Add the source node type to the set of input node types
            self.in_types.add(curr_n_src)
            # Add the destination node type to the set of output node types
            self.out_types.add(curr_n_dest)

        # Convert sets to lists for consistency
        self.etypes = list(self.etypes)
        self.in_types = list(self.in_types)
        self.out_types = list(self.out_types)

        # Check if the filtering resulted in an empty set of edge types when not allowed
        if not allow_empty and len(self.etypes) == 0:
            raise ValueError(
                f"No nodes in the graph qualify for e_name={e_name}, n_dest={n_dest}"
            )
