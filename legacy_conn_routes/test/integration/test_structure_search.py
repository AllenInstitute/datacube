import json
import os

import requests
import pytest
import pandas as pd


def test_query(structure_query, compare_responses):
    (expected, obtained), query = structure_query
    compare_responses(expected, obtained)