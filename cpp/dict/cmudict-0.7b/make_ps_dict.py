#!/usr/bin/env python3
"""
Convert CMU Pronouncing Dictionary to PocketSphinx format.

This script strips stress markers (0, 1, 2) from phonemes and deduplicates
pronunciations that become identical after stress removal.

Example:
    interest    IH1 N T R AH0 S T     ->  interest    IH N T R AH S T
    interest(2) IH1 N T R IH0 S T     ->  interest(2) IH N T R IH S T
"""

import argparse
import re
import sys
from collections import defaultdict
from typing import TextIO, Dict, List


def strip_stress(phoneme: str) -> str:
    """
    Remove stress markers from a phoneme.

    CMUdict uses digits 0, 1, 2 at the end of vowel phonemes to indicate stress:
        0 = no stress
        1 = primary stress
        2 = secondary stress

    Args:
        phoneme: A phoneme string (e.g., 'AH0', 'T', 'IH1')

    Returns:
        The phoneme without stress markers (e.g., 'AH', 'T', 'IH')
    """
    return re.sub(r'[012]$', '', phoneme)


def parse_cmudict_line(line: str) -> tuple[str, List[str]] | None:
    """
    Parse a single line from CMUdict.

    Args:
        line: A line from the dictionary file

    Returns:
        Tuple of (base_word, phonemes) or None if line should be skipped

    Examples:
        "hello HH AH0 L OW1" -> ("hello", ["HH", "AH0", "L", "OW1"])
        "world(2) W ER1 L D" -> ("world", ["W", "ER1", "L", "D"])
        ";;; comment" -> None
    """
    line = line.strip()

    # Skip empty lines and comments
    if not line or line.startswith(';;;'):
        return None

    # Remove inline comments
    if '#' in line:
        line = line.split('#')[0].strip()

    # Split into word and phonemes
    parts = line.split()
    if len(parts) < 2:
        return None

    word_with_variant = parts[0]
    phonemes = parts[1:]

    # Extract base word by removing variant marker like (2), (3), etc.
    base_word = re.sub(r'\(\d+\)$', '', word_with_variant)

    return base_word, phonemes


def convert_dict(infile: TextIO, outfile: TextIO) -> int:
    """
    Convert CMUdict to PocketSphinx format by stripping stress markers.

    The function:
    1. Reads all pronunciations from the input
    2. Strips stress markers from phonemes
    3. Groups pronunciations by base word
    4. Removes duplicate pronunciations (same after stress removal)
    5. Writes sorted output with proper variant numbering

    Args:
        infile: Input file stream (CMUdict format)
        outfile: Output file stream (PocketSphinx format)

    Returns:
        Number of entries written
    """
    # Dictionary to store unique pronunciations for each word
    # Key: base word, Value: list of unique stress-free pronunciations
    word_pronunciations: Dict[str, List[str]] = defaultdict(list)

    # Read and process all lines
    for line in infile:
        result = parse_cmudict_line(line)
        if result is None:
            continue

        base_word, phonemes = result

        # Strip stress from all phonemes
        phonemes_no_stress = [strip_stress(p) for p in phonemes]
        pronunciation = ' '.join(phonemes_no_stress)

        # Only keep unique pronunciations (deduplication)
        if pronunciation not in word_pronunciations[base_word]:
            word_pronunciations[base_word].append(pronunciation)

    # Write output in sorted order
    entries_written = 0
    for base_word in sorted(word_pronunciations):
        pronunciations = word_pronunciations[base_word]

        for idx, pronunciation in enumerate(pronunciations):
            # First pronunciation has no suffix, subsequent ones get (2), (3), etc.
            if idx == 0:
                word_variant = base_word
            else:
                word_variant = f"{base_word}({idx + 1})"

            print(f"{word_variant} {pronunciation}", file=outfile)
            entries_written += 1

    return entries_written


def main() -> None:
    """
    Main entry point for the script.

    Parses command-line arguments and orchestrates the conversion process.
    """
    parser = argparse.ArgumentParser(
        description="Convert CMUdict to PocketSphinx format (strip stress markers)",
        epilog="Example: python make_ps_dict.py cmudict.dict -o pocketsphinx.dict"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="cmudict.dict",
        help="input CMUdict file (default: cmudict.dict)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="output file (default: stdout)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="print progress information to stderr"
    )

    args = parser.parse_args()

    # Log input file if verbose
    if args.verbose:
        print(f"Reading from: {args.input}", file=sys.stderr)

    # Read input and write output
    try:
        with open(args.input, 'r', encoding='latin-1') as infile:
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as outfile:
                    count = convert_dict(infile, outfile)
            else:
                count = convert_dict(infile, sys.stdout)

        # Log success if verbose
        if args.verbose:
            print(f"Successfully wrote {count:,} entries", file=sys.stderr)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found", file=sys.stderr)
        sys.exit(1)
    except IOError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
