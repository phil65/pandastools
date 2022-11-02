import pandas as pd
import scipy.fftpack
import scipy.io
import scipy.io.arff


@pd.api.extensions.register_dataframe_accessor("pt_scipy")
class DataFrameAccessor:
    def __init__(self, parent):
        self._obj = parent

    def fft(self):
        ds = self._obj
        ds.index = ds.index.pt.index_to_secs()
        sample_rate = ds.index[1] - ds.index[0]
        # TODO: transform seems to give different results than apply, dunno why atm...
        ds = ds.transform(lambda x: scipy.fftpack.rfft(x.to_numpy(), len(ds.index)))
        ds.index = scipy.fftpack.rfftfreq(len(ds.index), sample_rate)
        ds = ds.abs()
        return ds

        # @staticmethod
        # def load_matlab(path):
        #     mat = scipy.io.loadmat(str(path))
        # mdata = mat["measuredData"]
        # ndata = {n: mdata[n][0, 0] for n in mdata.dtype.names}
        # cols = [n for n, v in ndata.items() if v.size == ndata["numIntervals"]]
        # ds = pd.DataFrame(data=np.concatenate([ndata[c] for c in cols], axis=1),
        #                   index=[datetime.datetime(*ts) for ts in ndata["timestamps"]],
        #                   columns=cols)
        # ds = pd.DataFrame(np.hstack((mat['X'], mat['y'])))
        # return ds

    @staticmethod
    def load_arff(path):
        data, meta = scipy.io.arff.loadarff(str(path))
        ds = pd.DataFrame(data=data, columns=meta.names())
        for i, t in enumerate(meta.types()):
            if t == "nominal":
                ds.iloc[:, i] = ds.iloc[:, i].str.decode("utf-8").astype("category")
        return ds


if __name__ == "__main__":
    import pydataset

    import pandastools  # type: ignore  # noqa: F401

    test = pydataset.data("Formaldehyde")
    result = test.pt_scipy.fft()
    print(result)
