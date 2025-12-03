import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path2topo",
        type=str,
        default="/home/ubuntu/src/rlearn/graphs/topo_generic.csv",
        help="Path to the adjacency list describing the topology",
    )
    parser.add_argument(
        "--path2pos",
        type=str,
        default="/home/ubuntu/src/rlearn/graphs/pos_generic.csv",
        help="Path to the positions of the nodes in the topology",
    )
    parser.add_argument(
        "--anodes",
        default=["s3_10", "s1_7", "s2_9"],
        nargs="+",
        help="List of nodes that attack the network",
    )
    parser.add_argument(
        "--critserver",
        type=str,
        default="d3_1",
        help="The node that functions as the critical server",
    )
    parser.add_argument(
        "--optserver",
        default=["d3_2", "d4_1"],
        nargs="+",
        help="List of nodes that act as migration locations",
    )
    parser.add_argument(
        "--attmode",
        type=int,
        default=0,
        help="Modus operandi of the attacker, {0,1,2} default [0]",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        help="Modus operandi of the net: 'lan', 'wifi' and 'none'; default 'lan'",
    )
    parser.add_argument(
        "--path2tar",
        type=str,
        default="/home/ubuntu/src/rlearn/networks/targetmodel1.pt",
        help="Path where the nn target model is stored",
    )
    parser.add_argument(
        "--train",
        type=int,
        default=0,
        help="No train [0], train a new neural network with specified attackers [1], train a pre-trained neural network [2], train a new neural network with ith all available attackers in the topology [3], default [0]",
    )
    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        help="No eval [0], evaluate a neural network with specified attackers [1], evaluate a neural network with all available attackers in the topology [2], default [0]",
    )
    parser.add_argument("--display", type=int, default=1, help="show network pictures (1,0) [1]")
    args = parser.parse_args()
    return args
