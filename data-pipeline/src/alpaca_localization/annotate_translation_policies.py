#!/usr/bin/env python3
"""Entrypoint for Alpaca translation-policy annotation.

The shell runner keeps calling this module, while the implementation lives in
the translation_policy_annotation package.
"""

from alpaca_localization.translation_policy_annotation.cli import main


if __name__ == "__main__":
    main()
