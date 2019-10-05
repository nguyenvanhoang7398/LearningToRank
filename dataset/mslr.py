import numpy as np
import os
import pandas as pd


class MSLR(object):
    def __init__(self, train_file, dev_file, test_file=None, batch_size=32):
        self.batch_size = batch_size
        self.train_file = train_file
        self.dev_file = dev_file
        if test_file is None:
            print("Test file is not provided, use dev file instead")
            self.test_file = dev_file
        else:
            self.test_file = test_file

    def load_data(self):
        train_loader = DataLoader(self.train_file, self.batch_size)
        df_train = train_loader.load()
        dev_loader = DataLoader(self.dev_file, self.batch_size)
        df_dev = dev_loader.load()
        test_loader = DataLoader(self.test_file, self.batch_size)
        df_test = test_loader.load()
        return train_loader, df_train, dev_loader, df_dev, test_loader, df_test


class DataLoader(object):
    def __init__(self, input_path, batch_size):
        self.input_path = input_path
        self.batch_size = batch_size
        self.df = None
        self.num_features = None
        self.num_pairs = None
        self.num_sessions = None

    def get_num_pairs(self):
        if self.num_pairs is not None:
            return self.num_pairs
        self.num_pairs = 0
        for _, Y in self.generate_batch_per_query(self.df):
            Y = Y.reshape(-1, 1)
            pairs = Y - Y.T
            pos_pairs = np.sum(pairs > 0, (0, 1))
            neg_pairs = np.sum(pairs < 0, (0, 1))
            assert pos_pairs == neg_pairs
            self.num_pairs += pos_pairs + neg_pairs
        return self.num_pairs

    def get_num_sessions(self):
        return self.num_sessions

    def generate_batch_per_query(self, df=None):
        if df is None:
            df = self.df
        qids = df.qid.unique()
        np.random.shuffle(qids)
        for qid in qids:
            df_qid = df[df.qid == qid]
            yield df_qid[['{}'.format(i) for i in range(1, self.num_features + 1)]].values, df_qid.rel.values

    def generate_query_pairs(self, df, qid):
        df_qid = df[df.qid == qid]
        rels = df_qid.rel.unique()
        x_i, x_j, y_i, y_j = [], [], [], []
        for r in rels:
            df1 = df_qid[df_qid.rel == r]
            df2 = df_qid[df_qid.rel != r]
            df_merged = pd.merge(df1, df2, on='qid')
            df_merged.reindex(np.random.permutation(df_merged.index))
            y_i.append(df_merged.rel_x.values.reshape(-1, 1))
            y_j.append(df_merged.rel_y.values.reshape(-1, 1))
            x_i.append(df_merged[['{}_x'.format(i) for i in range(1, self.num_features + 1)]].values)
            x_j.append(df_merged[['{}_y'.format(i) for i in range(1, self.num_features + 1)]].values)
        return np.vstack(x_i), np.vstack(y_i), np.vstack(x_j), np.vstack(y_j)

    def _load_mslr(self):
        df = pd.read_csv(self.input_path, sep=" ", header=None)
        df.drop(columns=df.columns[-1], inplace=True)
        self.num_features = len(df.columns) - 2
        self.num_pairs = None
        return df

    def generate_query_pair_batch(self, df):
        if df is None:
            df = self.df
        x_i_buf, y_i_buf, x_j_buf, y_j_buf = None, None, None, None
        qids = df.qid.unique()
        np.random.shuffle(qids)
        for qid in qids:
            x_i, y_i, x_j, y_j = self.generate_query_pairs(df, qid)
            if x_i_buf is None:
                x_i_buf, y_i_buf, x_j_buf, y_j_buf = x_i, y_i, x_j, y_j
            else:
                x_i_buf = np.vstack((x_i_buf, x_i))
                y_i_buf = np.vstack((y_i_buf, y_i))
                x_j_buf = np.vstack((x_j_buf, x_j))
                y_j_buf = np.vstack((y_j_buf, y_j))
            idx = 0
            while (idx + 1) * self.batch_size <= x_i_buf.shape[0]:
                start = idx * self.batch_size
                end = (idx + 1) * self.batch_size
                yield x_i_buf[start: end, :], y_i_buf[start: end, :], x_j_buf[start: end, :], y_j_buf[start: end, :]
                idx += 1

            x_i_buf = x_i_buf[idx * self.batch_size:, :]
            y_i_buf = y_i_buf[idx * self.batch_size:, :]
            x_j_buf = x_j_buf[idx * self.batch_size:, :]
            y_j_buf = y_j_buf[idx * self.batch_size:, :]

        yield x_i_buf, y_i_buf, x_j_buf, y_j_buf

    def generate_query_batch(self, df):
        idx = 0
        while idx * self.batch_size < df.shape[0]:
            r = df.iloc[idx * self.batch_size: (idx + 1) * self.batch_size, :]
            yield r.qid.values, r.rel.values, r[['{}'.format(i) for i in range(1, self.num_features + 1)]].values
            idx += 1

    def _parse_feature_and_label(self, df):
        for col in range(1, len(df.columns)):
            if ':' in str(df.iloc[:, col][0]):
                df.iloc[:, col] = df.iloc[:, col].apply(lambda x: x.split(":")[1])
        df.columns = ['rel', 'qid'] + [str(f) for f in range(1, len(df.columns) - 1)]

        for col in [str(f) for f in range(1, len(df.columns) - 1)]:
            df[col] = df[col].astype(np.float32)

        self.df = df
        self.num_sessions = len(df.qid.unique())
        return df

    def load(self):
        cache_output_path = "{}.cache".format(self.input_path)
        if os.path.exists(cache_output_path):
            print("Load cache df from {}".format(cache_output_path))
            self.df = pd.read_pickle(cache_output_path)
            self.num_features = len(self.df.columns) - 2
            self.num_pairs = None
            self.num_sessions = len(self.df.qid.unique())
        else:
            self.df = self._parse_feature_and_label(self._load_mslr())
            self.df.to_pickle(cache_output_path)
        return self.df
