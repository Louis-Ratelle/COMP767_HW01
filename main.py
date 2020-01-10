import argparse
import numpy as np
import random
import os
import sys
import string
import re
import pickle


from time import time
from datetime import datetime
from nltk.tag import hmm
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist, ConditionalProbDist, FreqDist
# from nltk.lm import Vocabulary

ENCODING = 'latin1'
SAVED_MODELS_FOLDER = './saved_models/'
NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = 197710
TOL = 1e0

PLAIN = 0
CIPHER = 1

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        """Convert string to boolean (in argparse context)"""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='Breaking ciphered texts with HMM')
    parser.add_argument('cipher_folder', type=str,
                        help='The folder where the cipher is located.')
    parser.add_argument('-laplace', action='store_true', default=False,
                        help='Uses Laplace smoothing.')
    parser.add_argument('-lm', action='store_true', default=False,
                        help='Improves training by using external language '
                        'model.')
    parser.add_argument('--load', type=str, default=None,
                        help='Filename of a .pickle pre-saved model saved in '
                        '{} folder. Please include the .pickle extension.'
                        .format(SAVED_MODELS_FOLDER))
    return parser.parse_args()


# #############################################################################
#
# Loading files
#
# #############################################################################


def load_data(cipher_folder):
    '''Loads files into a dataset. Dataset is a dictionary containing
    two keys: train and test. Each key has as values a matrix of size
    (Nsentences, 2). Where the zeroth column is the plain sentence and
    the first column is the corresponding ciphered text.'''
    actions = ['train', 'test']

    dataset = {}

    for action in actions:
        filename = action + '_plain.txt'
        try:
            plain = np.loadtxt(os.path.join('./', cipher_folder, filename),
                               dtype='str', encoding=ENCODING, delimiter='\n')
            print('File {} loaded.'.format(filename))
        except FileNotFoundError:
            print('File {} not found.'.format(os.path.join('./',
                                                           cipher_folder,
                                                           filename)))
            break

        filename = action + '_cipher.txt'
        try:
            cipher = np.loadtxt(os.path.join('./', cipher_folder, filename),
                                dtype='str', encoding=ENCODING, delimiter='\n')
            print('File {} loaded.'.format(filename))
        except FileNotFoundError:
            print('File {} not found.'.format(os.path.join('./',
                                                           cipher_folder,
                                                           filename)))
            break

        dataset[action] = np.c_[plain, cipher]

    return dataset


# #############################################################################
#
# Pre-processing
#
# #############################################################################

def generate_tuples(data):
    '''Generates an array with data.shape[0] elements. Each element i of the
    array is derived from sample i and contains a list of tuples associating
    each character from that sample to the corresponding ciphered character.

    data    : matrix of size (Nsamples, 2) containing strings.
    output  : array of size Nsamples containing lists of tuples.'''

    tuples = []
    for sample in range(data.shape[0]):
        tuples.append(list(zip(data[sample, CIPHER], data[sample, PLAIN])))

    return tuples


def str2list(sentences):
    '''Converts list of sentences into list of list of characters.

    sentences: list of sentences

    output: list of size len(sentences). Each element on the list is
            a list of characters in the respective sentence.'''
    for idx in range(len(sentences)):
        sentences[idx] = list(sentences[idx])
    return sentences

def list2tuples(sentences):
    '''Converts list of sentences to
    list (sentences) of list of bigram tuples'''

    # merges sentences
    sentences = ' '.join(sentences)

    # create list of bigrams
    # for example, 'the cat sat on the mat'
    # returns [('t', 'h'), ('h', 'e'), ('e', ' '), ..., ('a', 't')]
    return list(zip(sentences, sentences[1:]))

def heads(sentences):
    '''Returns a list with the first character (head) of each sentence
    in sentences.
    sentences: list of sentences
    output: list of characters'''
    
    return [sentence[0] for sentence in sentences]

# #############################################################################
#
# Model
#
# #############################################################################

def train_hmm(labelled_sequences, alphabet, estimator):
    '''Trains a hidden markov model and returns a 
    tagger object'''
    trainer = hmm.HiddenMarkovModelTrainer(states=alphabet,
                                           symbols=alphabet)

    model = trainer.train_supervised(labelled_sequences, estimator=estimator)

    return model, trainer


def train_lm(model, data, alphabet):
    '''Updates the model tagger transitions and priors using an
    external corpus and saves the model'''
    args = get_arguments()
    print('Using extended language model for training.')

    # number of senteces to use from the corpus
    # -1 for entire corpus
    num_sentences = -1

    sentences = get_corpus(alphabet, num_sentences)
    sentences = list(sentences)

    # merge original data and corpus
    sentences = sentences + list(data[:, PLAIN])

    # remove empty sentences
    sentences = [s for s in sentences if len(s) != 0]

    print('Training (extended language model with {:d} sentences)...'.format(len(sentences)))

    bigrams = list2tuples(sentences)

    # update transition matrix
    cfdist = ConditionalFreqDist(bigrams)
    transitions = ConditionalProbDist(cfdist, estimator, len(alphabet)**2)

    # update priors
    fdist = FreqDist(heads(sentences))
    priors = estimator(fdist, len(alphabet))

    # redefine tagger model
    model = hmm.HiddenMarkovModelTagger(symbols=model._symbols,
                                        states=model._states,
                                        transitions=transitions,
                                        outputs=model._outputs,
                                        priors=priors)

    # saves model
    filename = args.cipher_folder + '_lm'
    if args.laplace:
        filename += '_laplace'
    save(model, filename)

    return model


def estimator(fd, bins):
    # define estimator
    args = get_arguments()
    if args.laplace:
            return hmm.LidstoneProbDist(freqdist=fd, gamma=1, bins=bins)
    else:
            return hmm.MLEProbDist(freqdist=fd, bins=bins)

# #############################################################################
#
# External corpus
#
# #############################################################################

def get_corpus(alphabet, num_sentences=10):
    ''' Retrieves a list of num_sentences from the Brown corpus.
    The list is cleaned to display only characters contained in the alphabet

    alphabet: set of characters in the alphabet
    num_sentences: number of sentences to retrieve from corpus
                   (integer, default=10)
                   use -1 for entire corpus

    output: list of sentences'''

    # define pattern of characters not in the alphabet
    pattern = re.compile(r'[^{}]+'.format(''.join(alphabet)))

    # import num_sentences from Brown corpus
    sentences = brown.sents()[:num_sentences]
    sentences = np.array(sentences)

    for idx in range(len(sentences)):
        # merge list of words into a single sentence
        sentences[idx] = ' '.join(sentences[idx])

        sentences[idx] = sentences[idx].lower()

        # remove characters not in the alphabet
        sentences[idx] = pattern.sub('', sentences[idx])

    return sentences


# #############################################################################
#
# Main
#
# #############################################################################


def save(model, filename):
    f = open(os.path.join(SAVED_MODELS_FOLDER,
                          NOW + '_' + filename + '.pickle'), 'wb')
    pickle.dump(model, f)
    f.close()


def load(filename):
    filename = os.path.join(SAVED_MODELS_FOLDER, filename)
    try:
        f = open(filename, 'rb')
        model = pickle.load(f)
        f.close()
    except FileNotFoundError:
        print('Could not open file {}'.format(filename))
        sys.exit()

    return model


def train(data):
    '''Trains a new Hidden Markov model.

    args: arguments from the command line parser
    data: dataset containing list of sentences and its corresponding cipher
          where the zeroth column is the plain sentence and
          the first column is the corresponding ciphered text.

    output: trained model'''
    args = get_arguments()
    t0 = time()

    if args.laplace:
        print('Using Laplace smoothing.')

    gold_tuples = generate_tuples(data)

    # defines the alphabet
    alphabet = set(string.ascii_lowercase)
    alphabet.add(' ')   # adds space
    alphabet.add(',')   # adds comma
    alphabet.add('.')   # adds stop

    alphabet = list(alphabet)

    # train the Hidden Markov Model
    model, trainer = train_hmm(gold_tuples, alphabet, estimator)
    print('\nTraining accuracy: {:0.2f}'.format(model.evaluate(gold_tuples) * 100))

    # incorporating external corpus
    if args.lm:
        model = train_lm(model, data, alphabet)
        print('\nTraining accuracy with extended language model: {:0.2f}'.format(model.evaluate(gold_tuples) * 100))

    trn_time = time() - t0
    print("\nTrain time: {:0.3f}s".format(trn_time))

    return model


def test(model, data):
    '''Runs model on test data.

    model: pre-trained model
    data: dataset containing list of sentences and its corresponding cipher
          where the zeroth column is the plain sentence and
          the first column is the corresponding ciphered text.'''
    for sample in range(data.shape[0]):
        print(''.join(model.best_path(data[sample, CIPHER])))

    test_tuples = generate_tuples(data)
    model.test(test_tuples)


def main():

    # parses command line arguments
    args = get_arguments()

    # loads data
    print('Reading files from {} folder.'.format(args.cipher_folder))
    dataset = load_data(args.cipher_folder)
    gold_tuples = generate_tuples(dataset['train'])

    if args.load is not None:
        # load pre-saved model
        model = load(args.load)
        print('\nTraining accuracy with extended language model: {:0.2f}'.format(model.evaluate(gold_tuples) * 100))
    else:
        # train a new model
        model = train(dataset['train'])

    # runs model on test data
    test(model, dataset['test'])


if __name__ == '__main__':
    main()
