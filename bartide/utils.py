from typing import Generator, Tuple
from .config import logger
import os
import glob

__all__ = ["glob_files"]


def glob_files(
    directory: str,
    read1_pattern: str = "R1",
    read2_pattern: str = "R2",
    file_extension: str = "fastq.gz",
) -> Generator[Tuple[str, str], None, None]:
    """

    :param directory:
    :param read1_pattern:
    :param read2_pattern:
    :param file_extension:
    :return:
    """
    if os.path.isdir(directory):
        logger.debug("Directory found!")
    else:
        logger.error(f"Directory '{directory}' not found!")
        return None
    samples1 = sorted(glob.glob(f"{directory}/*{read1_pattern}.{file_extension}"))
    samples2 = sorted(glob.glob(f"{directory}/*{read2_pattern}.{file_extension}"))
    try:
        assert len(samples1) == len(samples2)
    except AssertionError:
        logger.error(
            "Different number of read1 and read2 files found. "
            "Please check that `read1_pattern` and `read2_pattern` parameters are correct. "
            "Also check that you have an even number of correctly labelled files in the directory."
        )
        return None
    else:
        if len(samples1) == 0:
            logger.error(
                "No samples found. Check that directory and file extension are correct. "
                "Also that `read1_pattern` and `read2_pattern` parameter are correct. "
                "For example, for miSeq runs `read1_pattern` and `read2_pattern` could "
                "look like: 'R1_001' and 'R2_001' respectively."
            )
            return None
        print(f"INFO: {len(samples1)} samples found")
        for r1, r2 in zip(samples1, samples2):
            yield r1, r2
