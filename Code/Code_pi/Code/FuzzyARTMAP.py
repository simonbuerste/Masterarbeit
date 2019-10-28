from functools import partial
import numpy as np
l1_norm = partial(np.linalg.norm, ord=1, axis=-1)

"""
Class definition of Fuzzy ARTMAP. All relevant member functions are implemented here.
"""


class FuzzyARTMAP(object):
    """
    Fuzzy ARTMAP

    A supervised version of FuzzyART
    """

    def __init__(self, alpha=0.25, gamma=0.01, rho=0.65, epsilon=0.001, n_classes=0, s=1.2):
        """

        :param alpha: learning rate [0,1]
        :param gamma: choice parameter > 0
        :param rho: vigilance parameter [0,1]
        :param epsilon: match tracking [-1,1]
        :param n_classes: Number of Classes (Necessary for the Category Layer)
        :param s: Parameter for thresholding with "Nothing I Know"-Concept
        """
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # choice parameter
        self.rho = rho  # vigilance
        self.epsilon = epsilon  # match tracking
        self.s = s  # Nothing I Know Concept Parameter

        self.w = None
        self.out_w = None
        self.n_classes = n_classes
        self.true_pos_train = 0

    def _init_weights(self, x, y):
        self.w = np.atleast_2d(x)
        self.out_w = np.zeros((1, self.n_classes))
        self.out_w[0, y] = 1

    def _add_category(self, x, y):
        self.w = np.vstack((self.w, x))
        self.out_w = np.vstack((self.out_w, np.zeros(self.n_classes)))
        self.out_w[-1, y] = 1

    def _match_category(self, x, y=None):
        if self.w is None:
            return -1, 0
        else:        
            _rho = self.rho
            _s = self.s
            
            # Calculate the fuzzy activations of Input Data with internal Representations
            fuzzy_weights = np.minimum(x, self.w)
            fuzzy_norm = l1_norm(fuzzy_weights)
            activations = fuzzy_norm / (self.gamma + l1_norm(self.w))
            # matching = fuzzy_norm / l1_norm(x)

            # Calculate matching-values with normed Dot Product (Cosine Similarity)
            # of Input with internal Representations
            w_normed = np.diag(np.sqrt(np.dot(self.w, np.transpose(self.w))))
            x_normed = np.sqrt(np.dot(x, x))
            dot_product = np.dot(self.w, x)
            matching = dot_product/(w_normed*x_normed)

            # For more than one internal Category representation, "Nothing I know" mechanism is used
            if (len(np.unique(self.out_w, axis=0)) > 1):
                rho = np.minimum(_s * np.mean(matching), 0.9)
                rho = np.maximum(rho, 0.75)
            else:
                rho = _rho

            threshold = matching > rho
            print("Threshold:{}".format(rho))
            
            # Check all matching values higher than the threshold
            while not np.all(threshold == False):
                y_ = np.argmax(activations * threshold.astype(int))

                if y is None or self.out_w[y_, y] == 1:
                    self.true_pos_train += 1
                    return y_, matching
                else:
                    # Match-Traking takes place here
                    rho = matching[y_] + self.epsilon
                    # self.rho = _rho + 0.01
                    matching[y_] = 0
                    threshold = matching > rho
            return -1, matching

    def train(self, x, y, epochs=1):
        """
        Training of the Fuzzy ARTMAP is done here for a defined number of epochs
        # Arguments
            :param x:       2d array of size (samples, features), where all features are in [0,1]
            :param y:       1d array of size (samples, ) containing the class label of each sample
            :param epochs:  number of training epochs, the training samples are shuffled after each epoch
        # Returns
            :return:        self
        """
        samples = x
        if self.n_classes == 0:
            self.n_classes = len(set(y))

        # Wieghts are initialized with training sample if no weights are available yet
        if self.w is None:
            self._init_weights(samples[0], y[0])

        idx = np.arange(len(samples), dtype=np.uint32)

        for epoch in range(epochs):
            self.true_pos_train = 0
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category, _ = self._match_category(sample, label)
                if category == -1:
                    # If Category is not known or not good enough known, new category node is created
                    self._add_category(sample, label)
                else:
                    print(category)
                    # If Category is known and threshold is reached, adapt current representation with sample
                    w = self.w[category]
                    # self.w[category] = self.alpha * np.minimum(sample, w) + (1 - self.alpha) * w
                    self.w[category] = self.alpha * sample + (1 - self.alpha) * w

            # print("Training Accuracy: {:.4f}".format(self.true_pos_train/len(samples)))
        return self

    def test(self, x):
        """
        Function for predict class label with current Fuzzy ARTMAP for the given input data
        # Arguments
            :param x: 2d array  of size (samples, features), where all features are in [0,1]
        # Returns
            :return: class label for each provided sample
        """
        samples = x
        labels = np.zeros(len(samples))
        
        for i, sample in enumerate(samples):
            category, matching = self._match_category(sample)
            if category != -1:
                # If Category known (Threshold reached) get the label of the representation
                labels[i] = np.argmax(self.out_w[category])
            else:
                # Else, "Nothing I know"-Mechanism is active and User has to give the Category Label
                # label = int(input("Kategorie unbekannt. Geben Sie die Kategorie ein: "))
                # if self.w is None:
                #     self._init_weights(sample, label)
                # else:
                #     self._add_category(sample, label)
                labels[i] = -1
            
            if self.out_w is None:
                return_labels = np.array(-1)
            else:
                return_labels = np.argmax(self.out_w, axis=1)
            
        return labels, matching, return_labels

    def consolidation(self):
        """
        Consolidation of a Fuzzy ARTMAP Instance is done here. All Representations of a specific class are merged into
        one representation. This can help to save memory and limit the number of representations
        # Arguments
            None
        # Returns
            :return: Consolidated Representation Layer. Weights from equal Classes are merged
        """
        unique_classes = np.unique(self.out_w, axis=0)
        out_w_consolidated = np.zeros_like(unique_classes, dtype=int)
        w_consolidated = np.zeros((unique_classes.shape[0], self.w.shape[1]))
        for i in range(self.n_classes):
            idx = np.argwhere(self.out_w[:, i] == 1)
            if idx.size != 0:
                w_consolidated[i, :] = np.mean(self.w[idx, :], axis=0)
                out_w_consolidated[i, i] = 1

        self.w = w_consolidated
        self.out_w = out_w_consolidated

        return self

    def melding(self, modul_b_additional):
        """
        Function for melding "self" with the given additional module B
        # Arguments
            :param modul_b_additional: Additional Module B which will be melded into self
        # Returns
            :return: Melded Representations and Category Layer of "self" with additional module B
        """
        self.w = np.append(self.w, modul_b_additional.w, axis=0)
        self.out_w = np.append(self.out_w, modul_b_additional.out_w, axis=0)

        return self
