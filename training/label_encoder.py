class LabelEncoder:

    def __init__(self):
        self.label_to_index = {}
        self.index_to_label = {}

    def fit(self, labels):

        unique_labels = sorted(set(labels))

        for index, label in enumerate(unique_labels):

            self.label_to_index[label] = index
            self.index_to_label[index] = label

    def encode(self, label):
        return self.label_to_index[label]

    def decode(self, index):
        return self.index_to_label[index]
