import json
import os

import requests
import pytest
import pandas as pd


def test_query(injection_coordinate_query, compare_responses):
    (expected, obtained), query = injection_coordinate_query
    compare_responses(expected, obtained)