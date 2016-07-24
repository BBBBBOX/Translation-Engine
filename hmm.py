import numpy as np
import sys

class HMM:
    """
    Attributes
    ----------
    A : numpy.ndarray
        State transition probability matrix
    B: numpy.ndarray
        State output probability matrix
    pi: numpy.ndarray
        Initial state probablity vector
    states: dict(int -> str)
        Maps state index in A/B/pi to the state name
    outputs: dict(str -> int)
        Maps words to their index in B
    """

    def __init__(self, A, B, pi, states, outputs):
        self.A = A
        self.B = B
        self.pi = pi
        self.states = states
        self.outputs = outputs

    def _forward(self, obs_ixs):
        # N = number of states
        N = self.A.shape[0]
        T = len(obs_ixs)
        # a = alpha
        a = np.zeros((N,T))
        a[:,0] = self.pi * self.B[:, obs_ixs[0]]

        for t in range(1, T):
            for x in range(N):
                a[x,t] = a[:,t-1].dot(self.A[:,x]) * self.B[x, obs_ixs[t]]

        return a

    def _backward(self, obs_ixs):
        N = self.A.shape[0]
        T = len(obs_ixs)
        b = np.zeros((N,T))
        b[:,-1:] = 1

        for t in reversed(range(T-1)):
            for k in range(N):
                b[k,t] = np.sum(b[:,t+1] * self.A[k,:] * self.B[:, obs_ixs[t+1]])

        return b

    def recognize(self, obs_seq):
        obs_ixs = [self.outputs[obs] for obs in obs_seq]
        return np.sum(self._forward(obs_ixs)[:,-1])

    def state_path(self, obs_seq):
        obs_ixs = [self.outputs[obs] for obs in obs_seq]
        V, prev = self.viterbi(obs_ixs)
        last_state = np.argmax(V[:,-1])
        path = list(self.build_path(prev, last_state))

        path_names = map(lambda k : self.states[k], reversed(path))
        return V[last_state,-1], path_names

    def viterbi(self, obs_ixs):
        # Number of states
        N = self.A.shape[0]
        T = len(obs_ixs)
        # Points to index of previous max state
        prev = np.zeros((len(obs_ixs) - 1, N))
        # DP matrix containing max likelihood of state sequences
        V = np.zeros((N, T))
        A = self.pi * self.B[:,obs_ixs[0]]
        V[:,0] = self.pi * self.B[:,obs_ixs[0]]

        for t in range(1, T):
            for k in range(N):
                seq_probs = V[:,t-1] * self.A[:,k] * self.B[k, obs_ixs[t]]
                prev[t-1,k] = np.argmax(seq_probs)
                V[k,t] = np.max(seq_probs)

        return V, prev

    def build_path(self, prev, last_state):
        yield(last_state)
        for i in range(len(prev) - 1, -1, -1):
            yield(prev[i, last_state])
            last_state = prev[i, last_state]

    def optimize(self, obs_seq):
        obs_ixs = [self.outputs[obs] for obs in obs_seq]
        T = len(obs_ixs)
        N = self.A.shape[0]

        obs_mat = np.zeros((T, self.B.shape[1]))
        obs_mat[range(T),obs_ixs] = 1

        forw = self._forward(obs_ixs)
        back = self._backward(obs_ixs)

        obs_prob = np.sum(forw[:,-1])
        if obs_prob == 0:
            return None
        gamma = forw * back / obs_prob

        xi = np.zeros((T-1, N, N))

        for t in range(xi.shape[0]):
            o = obs_ixs[t + 1]
            # [t] keeps dimensions on slice
            xi[t,:,:] = self.A * forw[:,[t]] * self.B[:,o] * back[:,t+1] / obs_prob

        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        rows_to_keep =  0 + (gamma_sum == 0)

        gamma_sum[gamma_sum == 0] = 1.

        # The gamma sum for A is from 0:T-1. Minus by the last gamma
        next_A = np.sum(xi, axis=0) / (gamma_sum - gamma[:,[-1]])
        next_B = gamma.dot(obs_mat) / gamma_sum

        self.A = self.A * rows_to_keep + next_A
        self.B = self.B * rows_to_keep + next_B
        self.pi = gamma[:,0] / np.sum(gamma[:,0])

        return True

def write_hmm_file(hmm, file_path):
    with open(file_path, 'wb') as f:
        s = ' '.join([str(len(hmm.states)), str(len(hmm.outputs)), '5']) + '\n'
        f.write(s)
        states = map(hmm.states.get, sorted(hmm.states.keys()))
        vocab = sorted(hmm.outputs.keys(), key=hmm.outputs.get)
        f.write(' '.join(states))
        f.write("\n")
        f.write(' '.join(vocab))
        f.write("\na:\n")
        f.write(array_to_str(hmm.A))
        f.write("\nb:\n")
        f.write(array_to_str(hmm.B))
        f.write("\npi:\n")
        f.write(' '.join('{:.6f}'.format(x) for x in hmm.pi.tolist()))

def array_to_str(arr):
    # Convert to float
    arr = [' '.join(['{:.6f}'.format(x) for x in y]) for y  in arr.tolist()]
    return '\n'.join(arr)

def read_hmm_file(file_path):
    with open(file_path, 'r') as f:
        lengths = f.readline().strip().split()
        num_states = int(lengths[0])
        num_vocab = int(lengths[1])
        obs_len = lengths[2]

        states = dict(enumerate(f.readline().strip().split()))
        vocab = dict(map(reversed, enumerate(f.readline().strip().split())))

        # Skip the "b:" line
        f.readline()
        # Read state output emission probability matrix
        A_list = [f.readline().strip().split() for i in range(num_states)]
        A = np.array([map(float, x) for x in A_list])

         # Skip the "b:" line
        f.readline()
        # Read state output emission probability matrix
        B_list = [f.readline().strip().split() for i in range(num_states)]
        B = np.array([list(map(float, x)) for x in B_list])

        # Skip the "pi:" line
        f.readline()
        probs = map(float, f.readline().strip().split())
        pi = np.array(probs)

        return HMM(A, B, pi, states, vocab)

def read_obs_file(file_path):
    with open(file_path, 'r') as f:
        num_seq = int(f.readline().strip())
        obs_seqs = []
        for i in range(num_seq):
            # Skip sequence length line
            f.readline()
            obs_seqs.append(f.readline().strip().split())

        return obs_seqs

if __name__ == '__main__':
    hmm_file_path = sys.argv[1]
    obs_file_path = sys.argv[2]

    hmm = read_hmm_file(hmm_file_path)

    obs_seqs = read_obs_file(obs_file_path)
    for obs_seq in obs_seqs:
        print(hmm.recognize(obs_seq))

        hmm.optimize(obs_seq)
        print(hmm.recognize(obs_seq))
