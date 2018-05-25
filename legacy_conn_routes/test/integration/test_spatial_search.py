import json
import os

import requests
import pytest
import pandas as pd


def test_query(spatial_query, compare_responses):
    (expected, obtained), query = spatial_query
    compare_responses(expected, obtained)