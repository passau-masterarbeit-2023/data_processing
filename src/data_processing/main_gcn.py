from graph_conv_net.data_loading.data_loading import load_annotated_graph
from graph_conv_net.params.params import ProgramParams



def main(params: ProgramParams):
    
    # load data
    print("Loading data...")
    print("Annotated graph from: {0}".format(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH))
    #print("No annotated graph from: {0}".format(params.NO_ANNOTATION_GRAPH_DOT_GV_DIR_PATH))

    annotated_graph = load_annotated_graph(params.ANNOTATED_GRAPH_DOT_GV_DIR_PATH + "/Performance_Test_Performance_Test_7572-1650972667-heap.raw_dot.gv")

    # print data
    print("Annotated graph:")
    print(annotated_graph)

    import pygraphviz as pgv
    import networkx as nx
    from torch_geometric.utils import from_networkx
    import torch

    # Read the GraphViz file
    graph_viz = pgv.AGraph("example.dot")

    # Convert to a NetworkX graph
    nx_graph = nx.Graph(graph_viz)

    # Convert to PyTorch Geometric Data object
    data = from_networkx(nx_graph)

    # Add some dummy node features (e.g., all ones)
    data.x = torch.ones(data.num_nodes, 1)

    print(data)


if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)