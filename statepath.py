import sys
from hmm import read_hmm_file, read_obs_file

if __name__ == '__main__':
    hmm_file_path = sys.argv[1]
    obs_file_path = sys.argv[2]

    hmm = read_hmm_file(hmm_file_path)

    obs_seqs = read_obs_file(obs_file_path)
    for obs_seq in obs_seqs:
        obs_prob, path = hmm.state_path(obs_seq)
        result = str(obs_prob)
        if obs_prob != 0:
            result += ' ' + ' '.join(path)
        print(result)
