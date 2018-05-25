import json
import os

import requests
import pytest
import pandas as pd


def test_query(correlation_query, compare_responses):
    (expected, obtained), query = correlation_query
    compare_responses(expected, obtained)