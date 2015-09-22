from csv import writer, reader
from re import sub
from subprocess import call
from os import path, remove
from shutil import rmtree
from glob import glob
from inspect import isgenerator

# Authors: Chris Emmery
# References: Ramage, Hall, Nallapati, Manning (2009)
# License: BSD 3-Clause


class STMT:

    """
    Stanford Topic Modelling Toolbox Wrapper
    =====
    This is a wrapper Class around the Stanford Topic Modelling Toolbox. It
    assumes that you have your vector space in your code, and don't want to
    bother with the `csv -> scala -> java -> csv -> extract results` process.
    It therefore compresses all of this in a few class interactions. Basically,
    you create model by initiating it with a name, set the amount of epochs
    and memory as desired, and then start training and testing on data that
    you have in Python code. After, the class can handle extracting the correct
    results (even in sklearn format), as well as cleaning up once you're done.
    Some examples of this will be given below, more information can be found
    on https://cmry.github.io/2015/06/18/topbox/.

    Parameters
    -----
    name : string
        The name that will be appended to all the saved files. If you want to
        keep the trained model, this name can be used to load it back in.

    epochs : integer, optional, default 20
        The amount of iterations you want L-LDA to train and sample; if you
        run into some errors, it's a good idea to set this to 1 to save time
        whilst debugging.

    mem : integer, optional, default 7000
        The amount of memory (in MB) that the model will use. By default it
        assumes that you have 8G of memory, so it will account for 1G of os
        running. Should be comfortable; adjust if running into OutOfMemory
        errors though.

    keep : boolean, optional, default True
        If set to False, will remove the data and scala files after training,
        and will remove EVERYTHING after the resutls are obtained. This can
        be handy when running a quick topic model and save disk space. If
        you're running a big model and want to keep it after your session is
        done, it might be better to just leave it to True.

    Attributes
    -----
    dir : string
        Absolute path where the storage area of the topbox is located.

    Examples
    -----
    train = [['sports football', 'this talks about football, or soccer,
               with a goal and a ball'],
             ['sports rugby', 'here we have some document where we do a scrum
               and kick the ball'],
             ['music concerts', 'a venue with loud music and a stage'],
             ['music instruments', 'thing that have strings or keys, or
               whatever']]

    test = [['music', 'the stage was full of string things'],
            ['sports', 'we kick a ball around'],
            ['rugby', 'now add some confusing sentence with novel words what is
              happening']]

    import topbox

    stmt = topbox.STMT('test_model')
    stmt = topbox.STMT('test_model', epochs=400, mem=14000)

    train_labels, train_space = zip(*train)
    test_labels, test_space = zip(*test)

    stmt.train(train_space, train_labels)
    stmt.test(test_space, test_labels)

    y_true, y_score = stmt.results(test_labels, array=True)

    from sklearn.metrics import average_precision_score
    average_precision_score(y_true, y_score)

    Notes
    -----
    The code and scale examples are obtained from the Stanford website
    (http://nlp.stanford.edu/software/tmt/tmt-0.4/). Their code thusly exists
    in this repository under equal license. Please respect this.
    """

    def __init__(self, name, epochs=20, mem=7000, keep=True):
        self.dir = path.dirname(path.realpath(__file__))+'/box/'
        self.name = name
        self.keep = keep
        self.epochs = epochs
        self.mem = mem

    def boot(self, mod):
        """
        Boot script
        =====
        Alters the directories in the .scala files for running and testing
        L-LDA (depending on the `mod`). Uses a generic call on the .jar that
        STMT resides in.

        Parameters
        -----
        :mod: string
            Either 'test' or 'train' for swithing states.
        """
        self.scala(mod)
        call(["java", "-Xmx"+str(self.mem)+"m", "-jar",
              self.dir+"tmt-0.4.0.jar", self.dir+self.name+"_"+mod+".scala"])
        self.scala(mod, 1)

    def store(self, space, labels, vsp_type):
        """
        Data to csv storage
        =====
        Stores a given (sub)vectorspace to the .csv format that STMT works
        with. The space should be a dict where the key is a tuple with (int,
        str), where int is the index number and str the document its topic
        labels seperated by a whitespace. The value is your vector stored in
        a list.

        If you want to iteratively construct a space, provide a generator that
        will feed batches of the space.

        Parameters
        -----
        space : list
            The vector space; a list with text.

        labels : list
            List with labels where each index corresponds to the text in space.

        vps_type : string
            Either train or test as appendix for the filename.
        """
        csv_file = open("%s%s_%s.csv" % (self.dir, self.name, vsp_type), 'a')
        csv_writer = writer(csv_file)
        for i, zipped in enumerate(zip(labels, space)):
            line = [str(i+1), zipped[0], zipped[1]]
            csv_writer.writerow(line)
        csv_file.close()

    def regex(self, f, needle, rock):
        """
        File name replacer
        =====
        Function is used to flip the read object file (original .scale file)
        and write replaced cotents to this newly created file.

        Parameters
        -----
        f : string
            Contents of the original .scala file.

        needle : string
            String sequence to be replaced in the original .scala file.

        rock : string
            Basically the .read() contents of the original .scala file.
        """
        wf = f.replace('_', self.name+'_')
        f = f.replace('_', '')
        try:
            rf = open(wf, 'r')
        except IOError:
            rf = open(f, 'r')
        stack = sub(needle, rock, rf.read())
        rf.close()
        with open(wf, 'w') as wf:
            wf.write(stack)

    def scala(self, step, s=False):
        """
        Scala code replacer
        =====
        Handles the .scala text replacements. In the basefiles,the replace
        targets are `modelfile` by default. This can also be used to flip
        number of the iterations.

        Parameters
        -----
        :step: train or test
        :s:    indicates old to new replace by default
        """
        prep, std = 'maxIterations = ', ' 5'
        orig, new = 'modelfolder', self.dir+self.name+'_'+'train'  # train
        o_csv, n_csv = 'datafile.csv', self.dir+self.name+'_'+step+'.csv'
        f = self.dir+'_'+step+'.scala'
        self.regex(f, o_csv, n_csv) if not s else self.regex(f, n_csv, o_csv)
        self.regex(f, orig, new) if not s else self.regex(f, new, orig)
        self.regex(f, prep+std, prep+' '+str(self.epochs)) if \
            self.epochs else self.regex(f, prep+std, prep+std)

    def m_incidence(self, predicted_row, label_index, gold_standard):
        """
        Matrix to Incidence
        =====
        Extracts the probabilities from the .csvs, and generates an incidence
        vector based on the correct topic labels. If a value is 'NaN', it will
        be skipped (model might have crapped up somewhere). The result is a
        zipped matrix with tuple values giving (incidence, probability).

        Parameters
        -----
        predicted_row : list
            Predicted row in the .csv file.
        label_index : list
            Lookup list for topics on index number.
        gold_standard : list
            Lookup list for correct topics per document.

        Return
        -----
        vector : list of lists
            Incidence matrix with: list(list(tuple(incidence, probability))).
        """
        if 'NaN' in predicted_row:  # don't wanna return NaN
            return
        else:
            vector = [(1 if label_index[i] in gold_standard else 0,
                       float(predicted_row[i+1])) for i in
                      range(len(label_index))]
            return vector

    def get_scores(self, label_index, predicted_weights, true_labels):
        """
        Grab results
        =====
        Given the labelled and original file, retrieve for each
        vector: the correct label, ranks and probabilities. Get
        tuple vector, unzip it and add the incidence part to
        y_true and the probability part to y_score (these are
        sklearn arrays for evluation).

        Parameters
        -----
        label_index : list of tuples
            Enumerated list with topic indexes.
        predicted_weights : csv file containing label confidences
        true_labels : csv file containing original material

        Return
        ------
        y_true : list of integers
            Binary list (incidence matrix).

        y_score : list of floats
            Probabilities per topic.
        """
        y_true = []
        y_score = []
        for predicted_row, true_row in zip(predicted_weights, true_labels):
            gold_standard = true_row.lower().split()
            rank, prob = zip(*self.m_incidence(predicted_row, label_index,
                                               gold_standard))
            if 1 in rank:
                y_true.append(rank)
                y_score.append(prob)

        return y_true, y_score

    def to_array(self, y_true, y_score):
        """
        To sklean-ready array
        =====
        Converts the incidence matrix and its probabilites to a numpy format.
        Also cleans out columns that produce a sum of zeroes; this results in
        a division by zero error when determining recall. Dependencies are
        both numpu and scipy.

        Parameters
        -----
        y_true : list of integers
            Binary list (incidence matrix).

        y_score : list of floats
            Probabilities per topic.

        Return
        -----
        (y_true, y_score): numpy arrays
            Filtered and converted version of y_true and y_score input.
        """
        from collections import Counter
        import scipy
        import numpy as np

        def scan_empty(y_true):
            c = Counter()
            for x in y_true:
                for i, y in enumerate(x):
                    c[i] += y
            return [key for key, value in c.items() if value == 0]

        def lab_reduce(y_true, y_score):
            empty_indices = scan_empty(y_true)
            i = 0
            for k in empty_indices:
                y_true = scipy.delete(y_true, k-i, 1)
                y_score = scipy.delete(y_score, k-i, 1)
                i += 1
            return y_true, y_score

        return lab_reduce(np.asarray(y_true), np.asarray(y_score))

    def results(self, true_labels, array=False):
        """
        Results grabber
        =====
        Finds the predicted document topic distribution and label index for the
        model, then retrieves the actual labels from the original file and
        serves these to self.get_scores.

        labels : list
            The original set of labels per document

        array : boolean, optional, default False
            Returns a cleaned numpy array where a column cannot be all zeroes.
            Has numpy and scipy as dependencies; better handle this outside of
            the class if you do not want to work with those.

        Return
        -----
        y_true, y_score : list, list
            List of lists incidence matrix (binary) and list of lists document
            topic probabilities.
        """
        DTDA = 'document-topic-distributions-res'  # doctop file
        LIDX = '00000/label-index'                 # label index

        orf = open("%s%s_%s/%s.txt" % (self.dir, self.name,
                                       'train', LIDX), 'r')
        label_index = orf.read().lower().split('\n')[:-1]

        lbf = open("%s%s_%s/%s_%s-%s.csv" % (self.dir, self.name, 'train',
                                             self.name, 'test', DTDA), 'r')
        predicted_weights = reader(lbf)

        y_true, y_score = self.get_scores(label_index, predicted_weights,
                                          true_labels)

        lbf.close()
        orf.close()

        if array:
            y_true, y_score = self.to_array(y_true, y_score)

        self.cleanup(step='results')
        return y_true, y_score

    def cleanup(self, all=False, step=False):
        """
        Cleanup module
        =====
        If the user wants the trained model to be kept, it will only remove the
        .csvs and wordcounts. Otherwise, it also dumps the fully trained model
        in self.train.

        Parameters
        -----
        all : bool, optional, default False
            Can be used to remove ALL files from box.

        step : bool, optional, default False
            Indicates the step so that it will keep the compressed and model
            files.
        """
        pattern = self.name+'_*' if not all else '*_*'
        files = glob(self.dir+pattern)
        for f in files:
            if not self.keep and step != 'results':
                rmtree(f) if '.' not in f else remove(f)
            else:
                remove(f) if '.' in f and '.gz' not in f else None

    def run(self, space, labels, step):
        """
        Main runner
        =====
        Checks if the given space is given in a generator for batching, writes
        it out to a csv with self.store, then self.boot-s the model in either
        train or test mode. If it's in test, it will return the results so that
        self,results does not have to be used.

        Parameters
        -----
        space : list
            The vector space; a list with text.

        labels : list
            List with labels where each index corresponds to the text in space.

        step : str
            Either test or train.
        """
        if not isgenerator(space):
            space = [space]
            labels = [labels]
        for batch_space, batch_labels in zip(space, labels):
            self.store(batch_space, batch_labels, step)
        space, labels = None, None
        self.boot(step)
        self.cleanup()

    def train(self, space, labels):
        """
        Sugar train
        =====
        Will train a previously untrained STMT instance on the given
        vectorspace. Please check the store function for space requirements.
        Can accept a generator for both space and labels.

        Parameters
        -----
        space : list
            The vector space; a list with text.

        labels : list
            List with labels where each index corresponds to the text in space.
        """
        self.run(space, labels, 'train')

    def test(self, space, labels):
        """
        Sugar test
        =====
        Will test a previously trained STMT instance on the given vectorspace.
        Please check the store function for space requirements.
        Can accept a generator for both space and labels.

        Parameters
        -----
        space : list
            The vector space; a list with text.

        labels : list
            List with labels where each index corresponds to the text in space.
        """
        self.run(space, labels, 'test')
