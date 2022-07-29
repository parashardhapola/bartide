import gzip
import re
import editdistance
from tqdm import tqdm
import pandas as pd
from typing import Generator, Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from .config import logger

plt.rcParams["svg.fonttype"] = "none"


class BarcodeExtractor:
    def __init__(
        self,
        read1_file: str,
        read2_file: str,
        left_flank: Optional[str] = None,
        right_flank: Optional[str] = None,
        barcode_length: Optional[int] = None,
        max_dist: int = 3,
        max_read_length: int = 200,
        disable_progress_bar: bool = False,
    ):
        """
        Extract barcodes from paired-end FASTQ files. The files can be gunzip.

        :param read1_file: Read 1 FASTQ file
        :param read2_file: Read 2 FASTQ file
        :param left_flank: Left primer sequence
        :param right_flank: Right primer sequence
        :param barcode_length: Length of the barcode sequence
        :param max_dist: Maximum allowed distance between barcode sequence from match pair.
        :param max_read_length: Maximum read length
        :param disable_progress_bar: If True, then progress bars are not shown
        """
        self.r1Fn = read1_file
        self.r2Fn = read2_file
        self.leftFlank = left_flank
        self.rightFlank = right_flank
        self.barcodeLength = barcode_length
        self.maxDist = max_dist
        self.maxReadLength = max_read_length
        self.nucComp: Optional[pd.Series] = None
        self.rawCounts: Optional[pd.Series] = None
        self.disablePb = disable_progress_bar

    @staticmethod
    def _read_fastq(fn: str) -> Generator[str, None, None]:
        n = 1
        if fn.endswith(".gz"):
            to_str = lambda x: x.decode("UTF-8").rstrip("\n")
            handle = gzip.open(fn)
        else:
            to_str = lambda x: x.rstrip("\n")
            handle = open(fn)
        for line in handle:
            if n % 2 == 0 and n % 4 != 0:
                yield to_str(line)
            n += 1
        handle.close()

    def identify_flanks(
        self,
        max_frac_threshold: float = 0.6,
        flank_size: int = 6,
        n_rows: Optional[int] = None,
    ):
        """

        :param max_frac_threshold:
        :param flank_size:
        :param n_rows:
        :return:
        """
        nuc_comp = {x: {x: 0 for x in "ATGCN"} for x in range(self.maxReadLength)}
        n = 0
        for seq1 in tqdm(
            self._read_fastq(self.r1Fn),
            desc="Identifying flank sequences",
            disable=self.disablePb,
        ):
            for p, i in enumerate(seq1):
                nuc_comp[p][i] += 1
            n += 1
            if n_rows is not None and n > n_rows:
                break
        nuc_comp = pd.DataFrame(nuc_comp).T.sort_index()
        ncs = nuc_comp.sum(axis=1).values
        length_thresh = np.where(ncs > ncs[0] / 2)[0].max()
        nuc_comp = nuc_comp[:length_thresh]
        frac = nuc_comp.max(axis=1) / nuc_comp.sum(axis=1)
        p = np.where(frac < max_frac_threshold)[0]
        s, e = p[0], p[-1]
        if p.shape[0] == e - s + 1:
            self.leftFlank = "".join(nuc_comp[s - flank_size : s].idxmax(axis=1).values)
            self.rightFlank = "".join(
                nuc_comp[e + 1 : e + flank_size + 1].idxmax(axis=1).values
            )
            self.barcodeLength = len(p)
        else:
            logger.error(
                "Flank detection failed due to non-contiguous variable base composition "
            )
        self.nucComp = nuc_comp

    @staticmethod
    def _rev_comp(seq: str) -> str:
        rev_map = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        return "".join([rev_map[x] for x in seq[::-1]])

    def _extract_seq_with_flanks(
        self, seq: str, up_seq: str, down_seq: str
    ) -> Optional[str]:
        """

        :param seq:
        :param up_seq:
        :param down_seq:
        :return:
        """
        up_pos = [x.start() for x in re.finditer(up_seq, seq)]
        down_pos = [x.start() for x in re.finditer(down_seq, seq)]
        adjust_start = len(up_seq)
        barcode_len = self.barcodeLength + adjust_start
        if len(up_pos) > 0 and len(down_pos) > 0:
            valid_comb = []
            for i in up_pos:
                for j in down_pos:
                    if j - i == barcode_len:
                        valid_comb.append((i + adjust_start, j))
            if len(valid_comb) == 1:
                return seq[valid_comb[0][0] : valid_comb[0][1]]
        return None

    def _stream_reads(self) -> Generator[Tuple[str, str], None, None]:
        for seq1, seq2 in zip(self._read_fastq(self.r1Fn), self._read_fastq(self.r2Fn)):
            yield seq1, seq2
        return None

    def count_barcodes(
        self,
    ) -> None:
        """

        :return:
        """
        if (
            self.leftFlank is None
            or self.rightFlank is None
            or self.barcodeLength is None
        ):
            logger.error(
                "Flank sequence(s) and/or barcode length is unknown. Please run `identify_flanks` "
                "to detect automatically or provide them manually when calling `BarcodesPE`"
            )
            return None
        counts = {}
        n, f1, f2 = 0, 0, 0
        max_len = 0
        nuc_comp = {x: {x: 0 for x in "ATGCN"} for x in range(self.maxReadLength)}
        for seq1, seq2 in tqdm(
            self._stream_reads(), disable=self.disablePb, desc="Counting barcodes"
        ):
            if len(seq1) > max_len:
                max_len = len(seq1)
            for p, i in enumerate(seq1):
                nuc_comp[p][i] += 1
            b1, b2 = (
                self._extract_seq_with_flanks(seq1, self.leftFlank, self.rightFlank),
                self._extract_seq_with_flanks(
                    seq2,
                    self._rev_comp(self.rightFlank),
                    self._rev_comp(self.leftFlank),
                ),
            )
            if b1 is not None and b2 is not None:
                b2 = self._rev_comp(b2)
                if editdistance.eval(b1, b2) <= self.maxDist:
                    if b1 not in counts:
                        counts[b1] = 0
                    counts[b1] += 1
                else:
                    f2 += 1
            else:
                f1 += 1
            n += 1

        f1 = 100 * f1 / n
        f2 = 100 * f2 / n
        f1 = "%.2f" % f1
        f2 = "%.2f" % f2
        logger.info(
            f"{n} sequences processed. {len(counts)} unique(uncorrected) barcodes found."
        )
        logger.warning(
            f"Unable to find barcodes in {f1}% reads. {f2}% reads had too many mismatches."
        )
        if max_len > self.maxReadLength:
            logger.warning(
                f"Maximum observed read length ({max_len}) is higher than provided ({self.maxReadLength}). "
                "The nucleotide composition table is likely to be truncated."
            )
        else:
            logger.info(f"Maximum observed read length is {max_len}")
            self.maxReadLength = max_len
        self.nucComp = pd.DataFrame(nuc_comp).T.sort_index()[: self.maxReadLength]
        self.rawCounts = pd.Series(
            dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))
        )

    def plot_composition(
        self,
        save_name: str = None,
        verts: List[int] = None,
        fig_size: Tuple[int, int] = (12, 3),
    ) -> None:
        """

        :param save_name:
        :param verts:
        :param fig_size:
        :return:
        """
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        self.nucComp.plot(kind="bar", stacked=True, ax=ax, cmap="Set3")
        if verts is not None:
            for i in verts:
                ax.axvline(i, color="k", lw=3.5, ls="--")
        ax.set_xlabel("Position in read")
        ax.set_ylabel("Nucleotide frequency")
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
            plt.close()
        else:
            plt.show()

    def plot_barcode_frequency(
        self, save_name: str = None, fig_size: Tuple[int, int] = (6, 5)
    ) -> None:
        """

        :param save_name:
        :param fig_size:
        :return:
        """
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        x = self.rawCounts.apply(np.log10).sort_values(ascending=False).reset_index()[0]
        x.plot(kind="line", ax=ax, lw=4)
        ax.set_xlabel("Barcodes sorted by frequency", fontsize=12)
        ax.set_ylabel("Log10 frequency", fontsize=12)
        plt.tight_layout()
        if save_name is not None:
            plt.savefig(save_name, dpi=300)
            plt.close()
        else:
            plt.show()
