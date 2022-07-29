import editdistance
from tqdm import tqdm
from typing import Optional, Dict
import nmslib
import pandas as pd


class SeqCorrect:
    def __init__(
        self,
        min_counts: int = 20,
        max_dist: int = 3,
        ann_k: Optional[int] = None,
        ann_m: int = 30,
        ann_ef_construction: int = 100,
        ann_post: int = 0,
        n_threads: int = 2,
        ann_ef_search: int = 100,
        disable_progress_bar: bool = False,
    ):
        """
        Class for identifying erroneous barcodes and merging their frequency with their correct version.

        :param min_counts: Minimum barcode frequency threshold. Barcodes with less than this value are removed. This
                           check is performed on corrected counts.
        :param max_dist: Minimum distance threshold when comparing barcodes. Barcodes with less than this value is
                         considered to be an erroneous version of a higher abundance barcode.
        :param ann_k: Number of neighbours in the KNN search.
        :param ann_m: ANN index construction parameter M
        :param ann_ef_construction: ANN index construction EF
        :param ann_post: ANN index construction post
        :param n_threads: ANN number of threads
        :param ann_ef_search: ANN query parameter EF
        :param disable_progress_bar: If True, then progress bars are not shown
        """

        if ann_k is None:
            self.annK = 2 * ann_m  # Same as the default behaviour of nmslib
        else:
            self.annK = ann_k
        self.minCounts = min_counts
        self.maxDist = max_dist
        self.indexParams = {
            "M": ann_m,
            "efConstruction": ann_ef_construction,
            "post": ann_post,
            "indexThreadQty": n_threads,
        }
        self.queryParams = {"efSearch": ann_ef_search}
        self.index = None
        self.correctedCounts: Optional[pd.Series] = None
        self.disablePb = disable_progress_bar

    def _build_index(self, raw_counts: Dict[str, int]) -> None:
        index = nmslib.init(
            method="hnsw",
            space="leven",
            data_type=nmslib.DataType.OBJECT_AS_STRING,
            dtype=nmslib.DistType.INT,
        )
        index.addDataPointBatch(list(raw_counts.keys()))
        index.createIndex(self.indexParams, print_progress=True)
        self.index = index

    def _query_index(self, raw_counts: Dict[str, int]) -> Dict[str, str]:
        bc_list = list(raw_counts.keys())
        self.index.setQueryTimeParams(self.queryParams)
        nbrs = self.index.knnQueryBatch(
            bc_list, k=self.annK, num_threads=self.indexParams["indexThreadQty"]
        )
        nm = {}
        for n, i in tqdm(
            enumerate(nbrs),
            total=len(nbrs),
            desc="Querying ANN index",
            disable=self.disablePb,
        ):
            a = bc_list[n]
            for j in i[0]:
                b = bc_list[j]
                if j == n:
                    continue
                if raw_counts[a] > raw_counts[b]:
                    continue
                d = editdistance.eval(a, b)
                if d > self.maxDist:
                    break
                if a in nm:
                    if raw_counts[b] > raw_counts[nm[a]]:
                        nm[a] = b
                else:
                    nm[a] = b
        return nm

    def _correct(
        self, raw_counts: Dict[str, int], neighbors: Dict[str, str]
    ) -> Dict[str, int]:
        bc_list = list(raw_counts.keys())
        cor_counts = {}
        for i in tqdm(bc_list, desc="Merging barcodes", disable=self.disablePb):
            if i in neighbors:
                a = neighbors[i]
            else:
                a = i
            if a not in cor_counts:
                cor_counts[a] = 0
            cor_counts[a] += raw_counts[i]

        clean_counts = {}
        for i in tqdm(
            cor_counts, desc="Filtering corrected barcodes", disable=self.disablePb
        ):
            if raw_counts[i] <= cor_counts[i] and cor_counts[i] > self.minCounts:
                clean_counts[i] = cor_counts[i]
        return clean_counts

    def run(self, raw_counts: pd.Series) -> None:
        """
        Get corrected and filtered barcodes.

        :param raw_counts: Row barcodes counts, in form of a dictionary.
        :return: Corrected barcode counts in form of a dictionary
        """
        counts = dict(raw_counts.to_dict())
        self._build_index(counts)
        neighbors = self._query_index(counts)
        self.correctedCounts = pd.Series(self._correct(counts, neighbors))

    def save_to_csv(self, file_name: str):
        """

        :param file_name:
        :return:
        """
        self.correctedCounts.to_csv(file_name, header=False)
