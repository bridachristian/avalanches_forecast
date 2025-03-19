# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:17:58 2025

@author: Christian
"""

import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "reanalysis",
    "variable": ["surface_solar_radiation_downwards"],
    "year": "2024",
    "month": [
        "01", "02", "03",
        "04"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "daily_statistic": "daily_maximum",
    "time_zone": "utc+01:00",
    "frequency": "1_hourly",
    "area": [46.4, 10.62, 46.3, 10.7]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
