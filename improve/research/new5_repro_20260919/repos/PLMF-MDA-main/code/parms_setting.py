import argparse

def settings():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed. Default is 0.')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')

    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parallel workers. Default is 0.')

    parser.add_argument('--pos_sample', default="data/MDS/pos_MDA_S.edgelist",
                        help='Path to data fold. e.g., MDS/pos_MDA_S.edgelist or MDR/pos_MDA_R.edgelist')

    parser.add_argument('--neg_sample', default="data/MDS/neg_MDA_S.edgelist",
                        help='Path to data fold. e.g., data/neg_MDA_S.edgelist or MDR/neg_MDA_R.edgelist')

    parser.add_argument('--mirna_file', default="data/MDS/miRNA_ID_S.xlsx",
                        help='Path to miRNA data file. e.g., MDS/miRNA_ID_S.xlsx or MDR/miRNA_ID_R.xlsx')

    parser.add_argument('--drug_file', default="data/MDS/drug_ID_S.xlsx",
                        help='Path to drug data file. e.g., MDS/drug_ID_S.xlsx or MDR/drug_ID_R.xlsx')


    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate. Default is 5e-4.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate. Default is 0.5.')

    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay. Default is 5e-4.')

    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size. Default is 32.')

    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of epochs to train. Default is 80.')

    # Model parameter settings
    parser.add_argument("--drug_input_dim", type=int, default=69,
                        help="Drug input dimension (number of atom features). Default is 69.")

    parser.add_argument("--drug_hidden_dim", type=int, default=128,
                        help="Drug hidden dimension. Default is 128.")

    parser.add_argument("--drug_feature_dim", type=int, default=128,
                        help="Drug feature output dimension. Default is 256.")

    parser.add_argument("--mirna_feature_dim", type=int, default=128,
                        help="miRNA feature output dimension. Default is 256.")


    args = parser.parse_args()

    return args