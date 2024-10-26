import torch.nn as nn

class Classification(nn.Module):
    def __init__(self, config, num_tasks, model=None):
        super(Classification, self).__init__()

        if config.model.n_layers < 2:
            raise ValueError("# layers must > 1.")
        self.graph_pooling = config.model.graph_pooling
        self.model = model
        self.hidden_dim = config.model.hidden_dim
        self.num_tasks = num_tasks

        self.graph_pred_linear = nn.Linear(self.hidden_dim, self.num_tasks)
        return

    def forward(self, *argv):
        if len(argv) == 3:
            x, pos, batch = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, pos, batch = data, data.pos, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.model(x, pos, batch)

        # Different kind of graph pooling
        if self.graph_pooling == "sum":
            graph_representation = node_representation.sum(1).view(node_representation.size(0), node_representation.size(-1))
        elif self.graph_pooling == "mean":
            graph_representation = node_representation.mean(1).view(node_representation.size(0), node_representation.size(-1))
        elif self.graph_pooling == "max":
            graph_representation = node_representation.max(1).view(node_representation.size(0), node_representation.size(-1))
        else:
            raise ValueError("Invalid graph pooling type.")

        output = self.graph_pred_linear(graph_representation)

        return output
