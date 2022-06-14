import editdistance
from tqdm.auto import tqdm
from typing import Optional, Dict
import nmslib


class SeqCorrect:
    def __init__(
        self,
        counts: Dict[str, int],
        min_counts: int = 20,
        ann_m: int = 30,
        ann_ef_search: int = 100,
        ann_ef_construction: int = 100,
        n_threads: int = 2,
        ann_post: int = 0,
        ann_k: Optional[int] = None,
    ):
        self.rawCounts = counts
        self.bc = list(self.rawCounts.keys())
        if ann_k is None:
            self.annK = 2 * ann_m  # Same as the default behaviour of nmslib
        else:
            self.annK = ann_k
        self.minCounts = min_counts
        self.indexParams = {
            "M": ann_m,
            "indexThreadQty": n_threads,
            "efConstruction": ann_ef_construction,
            "post": ann_post,
        }
        self.queryParams = {"efSearch": ann_ef_search}
        self.index = None
        self.nm = None

    def _build_index(self):
        index = nmslib.init(
            method="hnsw",
            space="leven",
            data_type=nmslib.DataType.OBJECT_AS_STRING,
            dtype=nmslib.DistType.INT,
        )
        index.addDataPointBatch(self.bc)
        index.createIndex(self.indexParams, print_progress=True)
        self.index = index

    def _query_index(self):
        self.index.setQueryTimeParams(self.queryParams)
        nbrs = self.index.knnQueryBatch(
            self.bc, k=self.annK, num_threads=self.indexParams["indexThreadQty"]
        )
        nm = {}
        for n, i in tqdm(enumerate(nbrs), total=len(nbrs)):
            for j in i[0]:
                if j == n:
                    continue
                if self.rawCounts[self.bc[n]] > self.rawCounts[self.bc[j]]:
                    continue
                d = editdistance.eval(self.bc[n], self.bc[j])
                if d > 3:
                    break
                if self.bc[n] in nm:
                    if self.rawCounts[self.bc[j]] > self.rawCounts[nm[self.bc[n]]]:
                        nm[self.bc[n]] = self.bc[j]
                else:
                    nm[self.bc[n]] = self.bc[j]
        self.nm = nm

    def _correct(self) -> Dict[str, int]:
        cor_counts = {}
        for i in tqdm(self.bc):
            if i in self.nm:
                a = self.nm[i]
            else:
                a = i
            if a not in cor_counts:
                cor_counts[a] = 0
            cor_counts[a] += self.rawCounts[i]

        clean_counts = {}
        for i in cor_counts:
            if self.rawCounts[i] <= cor_counts[i] and cor_counts[i] > self.minCounts:
                clean_counts[i] = cor_counts[i]
        return clean_counts

    def run(self) -> Dict[str, int]:
        self._build_index()
        self._query_index()
        return self._correct()
