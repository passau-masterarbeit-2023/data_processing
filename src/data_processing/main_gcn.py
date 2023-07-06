from graph_conv_net.data_loading.data_loading import load_annotated_graph, load_no_annotated_graph
from graph_conv_net.params.params import ProgramParams

def main(params: ProgramParams):
    
    # load data
    annotated_graph = load_annotated_graph(params.cli_args.args.annotated_graph_dot_gv_file_path)
    no_annotated_graph = load_no_annotated_graph(params.cli_args.args.no_annotated_graph_dot_gv_file_path)


if __name__ == "__main__":

    print("🚀 Running program...")
    params = ProgramParams()

    main(params)