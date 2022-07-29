import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from upsetplot import from_contents
from upsetplot import plot
from typing import List, Tuple
from glob import glob
import pathlib
from .config import logger


class BarcodeAnalyzer:
    def __init__(self, directory: str) -> None:
        """

        :param directory: Path to the directory that contains barcode CSV files
        """

        self.barcodes = {}
        for i in sorted(glob(f"./{directory}/*.csv")):
            name = pathlib.Path(i).name.replace(".csv", "")
            self.barcodes[name] = pd.read_csv(i, header=None, index_col=0)[1]

        if len(self.barcodes) == 0:
            logger.error(f"No barcodes found in the directory {directory}")
        else:
            self.barcodes = pd.DataFrame(self.barcodes).fillna(0).astype(int)
            self.barcodes.index.rename("barcodes", inplace=True)
            logger.info(f"{len(self.barcodes.columns)} barcode files found")

    def merge_groups(self, group_vec: List[str]) -> None:
        """

        :param group_vec:
        :return:
        """
        df = self.barcodes.copy().T
        df["group"] = group_vec
        df = df.groupby("group").min().T
        self.barcodes = df

    def make_subset(self, columns: List[str]) -> None:
        """

        :param columns:
        :return:
        """
        self.barcodes = self.barcodes[columns]
        self.barcodes = self.barcodes[self.barcodes.sum(axis=1) != 0]

    def calc_overlap(self, corrected: bool = False) -> pd.DataFrame:
        """

        :param corrected:
        :return:
        """
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
        """

        :return:
        """
        xdf = self.barcodes / self.barcodes.sum()
        w_o = {}
        for i in xdf:
            w_o[i] = {}
            for j in xdf:
                a = xdf[i]
                b = xdf[j]
                w_o[i][j] = ((a - b) ** 2).fillna(0).sum()
        w_o = pd.DataFrame(w_o)
        return 1 - w_o / w_o.max().max()

    def calc_percentage_overlap(self) -> pd.DataFrame:
        """

        :return:
        """
        overlap = self.calc_overlap()
        return 100 * overlap / overlap.sum()

    def plot_stacked(
        self,
        fig_size: Tuple[int, int] = (7, 5),
        save_name: str = None,
        rotation: float = 70,
    ) -> None:
        """

        :param fig_size:
        :param save_name:
        :param rotation:
        :return:
        """
        p_overlap = self.calc_percentage_overlap().cumsum().T
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        xind = list(range(p_overlap.shape[0]))
        for i in p_overlap.columns[::-1]:
            ax.bar(xind, p_overlap[i].values, label=i)
        ax.set_xticks(xind)
        ax.set_xticklabels(p_overlap.index, rotation=rotation)
        ax.legend(loc=(1.1, 0.5))
        ax.set_ylabel("% barcode overlap")
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        plt.show()

    def plot_upset(
        self, fig_size: Tuple[int, int] = (7, 5), save_name: str = None
    ) -> None:
        """

        :param fig_size:
        :param save_name:
        :return:
        """
        xdf = self.barcodes > 0
        xdf = from_contents({x: np.where(xdf[x])[0] for x in xdf})
        fig = plt.figure(figsize=fig_size)
        plot(xdf, sort_by="cardinality", fig=fig)
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        plt.show()

    def plot_weighted_heatmap(
        self,
        fig_size: Tuple[int, int] = (7, 7),
        cmap: str = "coolwarm",
        save_name: str = None,
        robust: bool = True,
    ) -> None:
        """

        :param fig_size:
        :param cmap:
        :param save_name:
        :param robust:
        :return:
        """
        sns.clustermap(
            self.calc_weighted_overlap(), cmap=cmap, figsize=fig_size, robust=robust
        )
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
        plt.show()

    def plot_overlap_heatmap(
        self,
        fig_size: Tuple[int, int] = (7, 7),
        cmap: str = "coolwarm",
        save_name: str = None,
        robust: bool = True,
    ) -> None:
        """

        :param fig_size:
        :param cmap:
        :param save_name:
        :param robust:
        :return:
        """
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
            plt.savefig(save_name, dpi=300)
        plt.show()
