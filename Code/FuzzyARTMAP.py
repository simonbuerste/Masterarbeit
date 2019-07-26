from functools import partial

import numpy as np

l1_norm = partial(np.linalg.norm, ord=1, axis=-1)


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
        """
        self.alpha = alpha  # learning rate
        self.beta = 1 - alpha
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
        _rho = self.rho
        _s = self.s

        fuzzy_weights = np.minimum(x, self.w)
        fuzzy_norm = l1_norm(fuzzy_weights)
        scores = fuzzy_norm / (self.gamma + l1_norm(self.w))
        # norms = fuzzy_norm / l1_norm(x)

        w_normed = np.diag(np.sqrt(np.dot(self.w, np.transpose(self.w))))
        x_normed = np.sqrt(np.dot(x, x))
        dot_product = np.dot(self.w, x)
        norms = dot_product/(w_normed*x_normed)

        if (len(norms) > 1) and (y is None):
            rho = np.minimum(_s * np.mean(norms), _s * _rho)
        else:
            rho = _rho

        threshold = norms > rho

        while not np.all(threshold == False):
            y_ = np.argmax(scores * threshold.astype(int))

            if y is None or self.out_w[y_, y] == 1:
                self.true_pos_train += 1
                return y_
            # elif 1 in self.out_w[:, y]:
            #     # convert the true class label to the category index inside ARTMAP
            #     category_idx = int(np.where(self.out_w[:, y] == 1)[0])
            #     return category_idx
            else:
                rho = norms[y_] + self.epsilon
                # self.rho = _rho + 0.01
                norms[y_] = 0
                threshold = norms > rho
        return -1

    def train(self, x, y, epochs=1):
        """
        :param x: 2d array of size (samples, features), where all features are in [0,1]
        :param y: 1d array of size (samples, ) containing the class label of each sample
        :param epochs: number of training epochs, the training samples are shuffled after each epoch
        :return: self
        """
        samples = x
        if self.n_classes == 0:
            self.n_classes = len(set(y))

        if self.w is None:
            self._init_weights(samples[0], y[0])

        idx = np.arange(len(samples), dtype=np.uint32)

        for epoch in range(epochs):
            self.true_pos_train = 0
            idx = np.random.permutation(idx)
            for sample, label in zip(samples[idx], y[idx]):
                category = self._match_category(sample, label)
                if category == -1:
                    self._add_category(sample, label)
                else:
                    w = self.w[category]
                    # self.w[category] = self.alpha * np.minimum(sample, w) + self.beta * w
                    self.w[category] = self.alpha * sample + self.beta * w

            print("Training Accuracy: {:.4f}".format(self.true_pos_train/len(samples)))
        return self

    def test(self, x):
        """
        :param x: 2d array  of size (samples, features), where all features are in [0,1]
        :return: class label for each provided sample
        """
        samples = x

        labels = np.zeros(len(samples))
        for i, sample in enumerate(samples):
            category = self._match_category(sample)
            if category != -1:
                labels[i] = np.argmax(self.out_w[category])
            else:
                print("{} tes Sample der Testdaten".format(i))
                label = int(input("Kategorie unbekannt. Geben Sie die Kategorie ein: "))
                self._add_category(sample, label)
                labels[i] = category
        return labels

    def consolidation(self):
        """

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
