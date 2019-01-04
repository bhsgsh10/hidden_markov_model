import sys, math, random
import numpy
import pdb

class HMM(object):
    # HMM model parameters #
    """
    Assume |X| state values and |E| emission values.
    initial: Initial belief probability distribution, list of size |X|
    tprob: Transition probabilities, size |X| list of size |X| lists;
           tprob[i][j] returns P(j|i), where i and j are states
    eprob: Emission probabilities, size |X| list of size |E| lists;
           eprob[i][j] returns P(j|i), where i is a state and j is an emission
    """
    def __init__(self, initial, tprob, eprob):
        self.initial = initial
        self.tprob = tprob
        self.eprob = eprob

    # Normalize a probability distribution
    def normalize(self, pdist):
        s = sum(pdist)
        for i in range(0,len(pdist)):
            pdist[i] = pdist[i] / s
        return pdist


    # Propagation (elapse time)
    """
    Input: Current belief distribution in the hidden state P(X_{t-1))
    Output: Updated belief distribution in the hidden state P(X_t)
    """
    def propagate(self, belief):
        final_list = []
        for i in range(len(belief)):
            x = 0
            sum_row = 0
            for j in range(len(self.tprob)):
                sum_row += belief[x] * self.tprob[j][i]
                x += 1
            final_list.append(sum_row)
        belief = final_list
        return belief

    # Observation (weight by evidence)
    """
    Input: Current belief distribution in the hidden state P(X_t),
           index corresponding to an observation e_t
    Output: Updated belief distribution in the hidden state P(X_t | e_t)  
    """
    def observe(self, belief, obs):

        observed_belief = []
        for i in range(len(belief)):
            observed_belief.append(belief[i] * self.eprob[i][obs])

        normalized_belief = self.normalize(observed_belief)
        belief = normalized_belief

        return belief

    # Filtering
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: Posterior belief distribution of the hidden state P(X_t | e_1, ..., e_t)
    """
    def filter(self, observations):
        # 1. start using the initial belief distribution available
        # 2. Use the observe function for each observation and keep updating the initial belief

        for i in range(len(observations)):
            self.initial = self.propagate(self.initial)
            self.initial = self.observe(self.initial, observations[i])
        return self.initial


    # Viterbi algorithm
    """
    Input: List of t observations in the form of indices corresponding to emissions
    Output: List of most likely sequence of state indices [X_1, ..., X_t]
    """
    def viterbi(self, observations):
        seq = []

        # maximize the probabilities first. Keep track of the state that leads to max
        timestep_state = {}

        timestep = 1
        for obsIndex in range(len(observations)):
            updated_belief = []
            max_row_value = 0
            for i in range(len(self.initial)):
                x = 0
                for j in range(len(self.tprob)):
                    max_row_value = max(max_row_value, self.initial[x] * self.tprob[j][i])
                    x += 1
                updated_belief.append(max_row_value)

            # Adding observation weighting
            updated_belief = self.observe(updated_belief, observations[obsIndex])
            timestep_state[timestep] = numpy.argmax(updated_belief)
            timestep += 1

        seq = list(timestep_state.values())

        return seq


# Functions for testing
# You should not change any of these functions
def load_model(filename):
    model = {}
    input = open(filename, 'r')
    i = input.readline()
    x = i.split()
    model['states'] = x[0].split(",")
    
    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['initial'] = [float(i) for i in y]

    input.readline()
    tprob = []
    for i in range(len(model['states'])):
        t = input.readline()
        x = t.split()
        y = x[0].split(",")
        tprob.append([float(i) for i in y])
    model['tprob'] = tprob

    input.readline()
    i = input.readline()
    x = i.split()
    y = x[0].split(",")
    model['emissions'] = dict(zip(y, range(len(y))))

    input.readline()
    eprob = []
    for i in range(len(model['states'])):
        e = input.readline()
        x = e.split()
        y = x[0].split(",")
        eprob.append([float(i) for i in y])
    model['eprob'] = eprob

    return model

def load_data(filename):
    input = open(filename, 'r')
    data = []
    for i in input.readlines():
        x = i.split()
        if x == [',']:
            y = [' ', ' ']
        else:
            y = x[0].split(",")
        data.append(y)
    observations = []
    classes = []
    for c, o in data:
        observations.append(o)
        classes.append(c)

    data = {'observations': observations, 'classes': classes}
    return data

def generate_model(filename, states, emissions, initial, tprob, eprob):
    f = open(filename,"w+")
    for i in range(len(states)):
        if i == len(states)-1:
            f.write(states[i]+'\n')
        else:
            f.write(states[i]+',')
    f.write('\n')

    for i in range(len(initial)):
        if i == len(initial)-1:
            f.write('%f\n'%initial[i])
        else:
            f.write('%f,'%initial[i])
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(states)):
            if j == len(states)-1:
                f.write('%f\n'%tprob[i][j])
            else:
                f.write('%f,'%tprob[i][j])
    f.write('\n')

    for i in range(len(emissions)):
        if i == len(emissions)-1:
            f.write(emissions[i]+'\n')
        else:
            f.write(emissions[i]+',')
    f.write('\n')

    for i in range(len(states)):
        for j in range(len(emissions)):
            if j == len(emissions)-1:
                f.write('%f\n'%eprob[i][j])
            else:
                f.write('%f,'%eprob[i][j])
    f.close()


def accuracy(a,b):
    total = float(max(len(a),len(b)))
    c = 0
    for i in range(min(len(a),len(b))):
        if a[i] == b[i]:
            c = c + 1
    return c/total

def test_filtering(hmm, observations, index_to_state, emission_to_index):
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]

    print('Short observation sequence:')
    print('   ', obs_short)
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_filter = hmm.filter(obs_indices_short)
    result_filter_full = hmm.filter(obs_indices)

    print('\nFiltering - distribution over most recent state given short data set:')
    for i in range(0, len(result_filter)):
        print('   ', index_to_state[i], '%1.3f' % result_filter[i])

    print('\nFiltering - distribution over most recent state given full data set:')
    for i in range(0, len(result_filter_full)):
        print('   ', index_to_state[i], '%1.3f' % result_filter_full[i])

def test_viterbi(hmm, observations, classes, index_to_state, emission_to_index):
    n_obs_short = 10
    obs_short = observations[0:n_obs_short]
    classes_short = classes[0:n_obs_short]
    obs_indices = [emission_to_index[o] for o in observations]
    obs_indices_short = obs_indices[0:n_obs_short]

    result_viterbi = hmm.viterbi(obs_indices_short)
    best_sequence = [index_to_state[i] for i in result_viterbi]
    result_viterbi_full = hmm.viterbi(obs_indices)
    best_sequence_full = [index_to_state[i] for i in result_viterbi_full]

    print('\nViterbi - predicted state sequence:\n   ', best_sequence)
    print('Viterbi - actual state sequence:\n   ', classes_short)
    print('The accuracy of your viterbi classifier on the short data set is', accuracy(classes_short, best_sequence))
    print('The accuracy of your viterbi classifier on the entire data set is', accuracy(classes, best_sequence_full))


# Train a new typo correction model on a set of training data (extra credit)
"""
Input: List of t observations in the form of string or other data type literals
Output: Dictionary of HMM quantities, including 'states', 'emissions', 'initial', 'tprob', and 'eprob' 
"""
def train(observations, classes):

    state_list = get_state_list(classes)
    emissions = get_emissions(observations)
    initial_belief = get_initial_belief(classes)
    transition_probabilities = get_trained_transtion_probabilities(classes)
    emission_probabilities = get_trained_emission_probabilities(classes, observations)


    return {'states': state_list, 'emissions': emissions, 'initial': initial_belief,
            'tprob': transition_probabilities, 'eprob': emission_probabilities}





def get_trained_transtion_probabilities(classes):
    k = 1
    # create a matrix of no. of states * no. of states and
    state_list = get_state_list(classes)
    transition_probabilities = []
    for i in range(len(state_list)):
        jlist = []
        for j in range(len(state_list)):
            jlist.append(k)
        transition_probabilities.append(jlist)


    state_index = {}
    counter = 0
    for state in state_list:
        state_index[state] = counter
        counter += 1

    for i in range(len(classes) - 1):
        #for j in range(i+1, len(classes)):

        transition_probabilities[state_index[classes[i]]][state_index[classes[i+1]]] += 1

    for i in range(len(state_list)):
        state = state_list[i]
        occurrences = classes.count(state)
        transition_probabilities[i] = [x/(occurrences + k*len(classes)) for x in transition_probabilities[i]]

    return transition_probabilities


def get_trained_emission_probabilities(classes, observations):
    k = 1
    # create a matrix of no. of states * no. of states and
    state_list = get_state_list(classes)
    emission_probabilities = []
    for i in range(len(state_list)):
        jlist = []
        for j in range(len(state_list)):
            jlist.append(k)
        emission_probabilities.append(jlist)


    state_index = {}
    counter = 0
    for state in state_list:
        state_index[state] = counter
        counter += 1

    for i in range(len(observations)):
        #for j in range(len(classes)):
        emission_probabilities[state_index[observations[i]]][state_index[classes[i]]] += 1

    for i in range(len(state_list)):
        state = state_list[i]
        occurrences = classes.count(state)
        emission_probabilities[i] = [x/(occurrences + k*len(observations)) for x in emission_probabilities[i]]

    return emission_probabilities


def get_state_list(classes):
    sorted_list = list(sorted(set(classes)))
    # _ should come at the end of the list. This is not a generic approach though
    if '_' in sorted_list:
        sorted_list.remove('_')
        sorted_list.append('_')

    return sorted_list

def get_emissions(observations):
    sorted_list = list(sorted(set(observations)))
    if '_' in sorted_list:
        sorted_list.remove('_')
        sorted_list.append('_')
    return sorted_list

def get_initial_belief(classes):
    initial_belief = []
    state_list = get_state_list(classes)
    marked = []
    class_freq = [(classes.count(i),marked.append(i))[0] for i in classes if i not in marked]
    class_freq  = [x/len(classes) for x in class_freq]
    '''
    for state in state_list:
        state_counter = 0
        for item in classes:
            if item == state:
                state_counter += 1
        state_belief = state_counter / len(classes)
        initial_belief.append(state_belief)
    '''
    return class_freq

def get_sorted_list(classes):
    sorted_classes = list(sorted(classes))
    if '_' in sorted_classes:
        sorted_classes.remove('_')
        sorted_classes.append('_')
    return sorted_classes


if __name__ == '__main__':
    # this if clause for extra credit training only
    if len(sys.argv) == 4 and sys.argv[1] == '-t':
        input = open(sys.argv[3], 'r')
        data = []
        for i in input.readlines():
            x = i.split()
            if x == [',']:
                y = [' ', ' ']
            else:
                y = x[0].split(",")
            data.append(y)

        observations = []
        classes = []
        for c, o in data:
            observations.append(o)
            classes.append(c)

        model = train(observations, classes)
        generate_model(sys.argv[2], model['states'], model['emissions'], model['initial'], model['tprob'], model['eprob'])
        exit(0)

    # main part of the assignment
    if len(sys.argv) != 3:
        print("\nusage: ./hmm.py [model file] [data file]")
        exit(0)

    model = load_model(sys.argv[1])
    data = load_data(sys.argv[2])
    new_hmm = HMM(model['initial'], model['tprob'], model['eprob'])
    # y = get_trained_transtion_probabilities(['a','b','g','d','g','a'])
    # test_filtering(new_hmm, data['observations'], model['states'], model['emissions'])
    test_viterbi(new_hmm, data['observations'], data['classes'], model['states'], model['emissions'])