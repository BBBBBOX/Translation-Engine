import sys
from hmm import read_obs_file, read_hmm_file, write_hmm_file

if __name__ == '__main__':
    hmm_file_path = sys.argv[1]
    obs_file_path = sys.argv[2]
    file_to_write = sys.argv[3]

    hmm = read_hmm_file(hmm_file_path)

    obs_seqs = read_obs_file(obs_file_path)
    for obs_seq in obs_seqs:
        result = hmm.recognize(obs_seq)
        if hmm.optimize(obs_seq) is None:
            print("Unable to optimize with \"{}\". P(O | lambda) = 0.".format(' '.join(obs_seq)))

        print(str(result) + " " + str(hmm.recognize(obs_seq)))
        write_hmm_file(hmm, file_to_write)
