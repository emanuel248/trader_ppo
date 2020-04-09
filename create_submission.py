import argparse
from model import ActorCritic


if __name__ == "__main__":
    mkdir('.', 'checkpoints')
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--weights", type=str, help="Weights to load")
    opts = parser.parse_args()

    # collect weights
    fc_layers = []

    # Get all hidden layers' weights
    for i in range(len(hidden_units)):
        fc_layers.extend([
            TrainNet.model.hidden_layers[i].weights[0].numpy().tolist(), # weights
            TrainNet.model.hidden_layers[i].weights[1].numpy().tolist() # bias
        ])

    # Get output layer's weights
    fc_layers.extend([
        TrainNet.model.output_layer.weights[0].numpy().tolist(), # weights
        TrainNet.model.output_layer.weights[1].numpy().tolist() # bias
    ])

    # Convert all layers into usable form before integrating to final agent
    fc_layers = list(map(
        lambda x: str(list(np.round(x, precision))) \
            .replace('array(', '').replace(')', '') \
            .replace(' ', '') \
            .replace('\n', ''),
        fc_layers
    ))
    fc_layers = np.reshape(fc_layers, (-1, 2))

    # Create the agent
    my_agent = '''def my_agent(observation, configuration):
        import numpy as np

    '''

    # Write hidden layers
    for i, (w, b) in enumerate(fc_layers[:-1]):
        my_agent += '    hl{}_w = np.array({}, dtype=np.float32)\n'.format(i+1, w)
        my_agent += '    hl{}_b = np.array({}, dtype=np.float32)\n'.format(i+1, b)
    # Write output layer
    my_agent += '    ol_w = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][0])
    my_agent += '    ol_b = np.array({}, dtype=np.float32)\n'.format(fc_layers[-1][1])

    my_agent += '''
        state = observation.board[:]
        state.append(observation.mark)
        out = np.array(state, dtype=np.float32)

    '''

    # Calculate hidden layers
    for i in range(len(fc_layers[:-1])):
        my_agent += '    out = np.matmul(out, hl{0}_w) + hl{0}_b\n'.format(i+1)
        my_agent += '    out = 1/(1 + np.exp(-out))\n' # Sigmoid function
    # Calculate output layer
    my_agent += '    out = np.matmul(out, ol_w) + ol_b\n'

    my_agent += '''
        for i in range(configuration.columns):
            if observation.board[i] != 0:
                out[i] = -1e7

        return int(np.argmax(out))
        '''
