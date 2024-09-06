# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:18:46 2024

@author: Christian
"""

"""This workflow is part of the document:
https://cds.climate.copernicus.eu/toolbox/doc/how-to/2_how_to_retrieve_time_series_and_extract_point_information/2_how_to_retrieve_time_series_and_extract_point_information.html#ht2
"""

# Initialise an application




import cdstoolbox as ct
@ct.application(title='Extract a time series and plot graph')
# Add output widget to the application
@ct.output.livefigure()
# Create a function that retrieves the Near Surface Air Temperature dataset, extracts data from a point and plots the line on the livefigure
def plot_time_series():

    # Retrieve a variable over a defined time range
    data = ct.catalogue.retrieve(
        'reanalysis-era5-single-levels',
        {
            'variable': '2m_temperature',
            'product_type': 'reanalysis',
            'year': [
                '2008'
            ],
            'month': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12'
            ],
            'day': [
                '01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30',
                '31'
            ],
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'grid': ['3', '3']
        }
    )

    # Select a location, defined by longitude and latitude coordinates
    data_point = ct.geo.extract_point(
        data,
        lon=75.0,
        lat=43.0
    )

    # Compute the daily mean for the selected data
    data_daily = ct.climate.daily_mean(data_point)

    # Show the result as a time-series on an interactive chart
    figure = ct.chart.line(data_daily)

    return figure
