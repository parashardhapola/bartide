import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from upsetplot import from_contents
from upsetplot import plot
from typing import List, Dict, Tuple


class BarcodeAnalyzer:
    def __init__(self, barcodes: Dict[str, Dict[str, int]]) -> None:
        self.barcodes = (
            pd.DataFrame({x: pd.Series(barcodes[x]) for x in barcodes})
            .fillna(0)
            .astype(int)
        )

    def merge_groups(self, group_vec: List[str]) -> None:
        df = self.barcodes.copy().T
        df["group"] = group_vec
        df = df.groupby("group").min().T
        self.barcodes = df

    def make_subset(self, columns: List[str]) -> None:
        self.barcodes = self.barcodes[columns]
        self.barcodes = self.barcodes[self.barcodes.sum(axis=1) != 0]

    def calc_overlap(self, corrected: bool = False) -> pd.DataFrame:
        overlap = {}
        df = self.barcodes.copy()
        df[df > 0] = 1
        for i in df.columns:
            overlap[i] = {}
            for j in df.columns:
                overlap[i][j] = ((df[i] + df[j]) == 2).sum()
                if corrected:
                    overlap[i][j] = overlap[i][j] / ((df[i] + df[j]) > 0).sum()
        return pd.DataFrame(overlap)

    def calc_weighted_overlap(self) -> pd.DataFrame:
        xdf = self.barcodes / self.barcodes.sum()
        w_o = {}
        for i in xdf:
            w_o[i] = {}
            for j in xdf:
                a = xdf[i]
                b = xdf[j]
                n = (xdf[[i, j]].sum(axis=1) != 0).sum()
                w_o[i][j] = ((a - b) ** 2).fillna(0).sum()
        w_o = pd.DataFrame(w_o)
        return 1 - w_o / w_o.max().max()

    def plot_stacked(
        self,
        fig_size: Tuple[int, int] = (7, 5),
        save_name: str = None,
        rotation: float = 70,
    ) -> None:
        overlap = self.calc_overlap()
        p_overlap = (100 * overlap / overlap.sum()).cumsum().T
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        xind = list(range(p_overlap.shape[0]))
        for i in p_overlap.columns[::-1]:
            ax.bar(xind, p_overlap[i].values, label=i)
        ax.set_xticks(xind)
        ax.set_xticklabels(p_overlap.index, rotation=rotation)
        ax.legend(loc=(1.1, 0.5))
        ax.set_ylabel("% barcode overlap")
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        plt.show()

    def plot_upset(
        self, fig_size: Tuple[int, int] = (7, 5), save_name: str = None
    ) -> None:
        xdf = self.barcodes > 0
        xdf = from_contents({x: np.where(xdf[x])[0] for x in xdf})
        fig = plt.figure(figsize=fig_size)
        plot(xdf, sort_by="cardinality", fig=fig)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()

    def plot_weighted_heatmap(
        self,
        fig_size: Tuple[int, int] = (7, 7),
        cmap: str = "coolwarm",
        save_name: str = None,
        robust: bool = True,
    ) -> None:
        sns.clustermap(
            self.calc_weighted_overlap(), cmap=cmap, figsize=fig_size, robust=robust
        )
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()

    def plot_overlap_heatmap(
        self,
        fig_size: Tuple[int, int] = (7, 7),
        cmap: str = "coolwarm",
        save_name: str = None,
        robust: bool = True,
    ) -> None:
        xdf = self.calc_overlap(corrected=True)
        xdf[xdf == 1] = 0
        sns.clustermap(
            self.calc_overlap(corrected=True),
            cmap=cmap,
            figsize=fig_size,
            vmax=xdf.max().max(),
            robust=robust,
        )
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()
