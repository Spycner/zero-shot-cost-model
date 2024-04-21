class PassDirection:
    """
    Defines a message passing step on the encoded query graphs.
    """

    def __init__(self, model_name, g, e_name=None, n_dest=None, allow_empty=False):
        """
        Initializes a message passing step.
        :param model_name: which edge model should be used to combine the messages
        :param g: the graph on which the message passing should be performed
        :param e_name: edges are defined by triplets: (src_node_type, edge_type, dest_node_type). Only incorporate edges
            in the message passing step where edge_type=e_name
        :param n_dest: further restrict the edges that are incorporated in the message passing by the condition
            dest_node_type=n_dest
        :param allow_empty: allow that no edges in the graph qualify for this message passing step. Otherwise this will
            raise an error.
        """
        self.etypes = set()
        self.in_types = set()
        self.out_types = set()
        self.model_name = model_name

        for curr_n_src, curr_e_name, curr_n_dest in g.canonical_etypes:
            if e_name is not None and curr_e_name != e_name:
                continue

            if n_dest is not None and curr_n_dest != n_dest:
                continue

            self.etypes.add((curr_n_src, curr_e_name, curr_n_dest))
            self.in_types.add(curr_n_src)
            self.out_types.add(curr_n_dest)

        self.etypes = list(self.etypes)
        self.in_types = list(self.in_types)
        self.out_types = list(self.out_types)
        if not allow_empty:
            assert (
                len(self.etypes) > 0
            ), f"No nodes in the graph qualify for e_name={e_name}, n_dest={n_dest}"
