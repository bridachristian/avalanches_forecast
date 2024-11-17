from pathlib import Path
import json


def codice_nivometeorologico():
    '''
    Dictionaries to map numerical codes with textual expression of 
    Codice Nivometeoologico per il rilievo giornaliero (mod.1 AINEVA)

    Returns
    -------
    data : dict
        a dictioary of dictioaries of codes.

    '''

    # https://artefacts.ceda.ac.uk/badc_datadocs/surface/code.html
    WW = {0: 'None of the following phenomena on the station at the time of observation',
          14: 'Precipitation within sight, not reaching the ground',
          36: 'Blizzard on the station',
          # Fog
          44: 'Fog or ice fog, sky visible',
          45: 'Fog or ice fog, sky invisible',
          # Rain
          60: 'Rain, not freezing, intermittent, slight ',
          61: 'Rain, not freezing, continuous, slight ',
          63: 'Rain, not freezing, continuous, moderate',
          65: 'Rain, not freezing, continuous, heavy',
          67: 'Rain, freezing, moderate or heavy',
          69: 'Rain or drizzle and snow, moderate or heavy',
          81: 'Rain shower(s), moderate or heavy',
          # Snow
          70: 'Intermittent fall of snowflakes, slight',
          71: 'Continuous fall of snowflakes, slight',
          73: 'Continuous fall of snowflakes, moderate',
          75: 'Continuous fall of snowflakes, heavy',
          84: 'Shower(s) of rain and snow, moderate or heavy',
          88: 'Shower(s) of snow pellets or small hail, with or without rain or rain and snow mixed - moderate or heavy',
          # Thunderstom
          95: 'Thunderstorm, slight or moderate, without hail, but with rain and/or snow',
          96: 'Thunderstorm, slight or moderate, with hail'}

    N = {0.0: '0/8 cloud cover',
         1.0: '1/8 cloud cover',
         2.0: '2/8 cloud cover',
         3.0: '3/8 cloud cover',
         4.0: '4/8 cloud cover',
         5.0: '5/8 cloud cover',
         6.0: '6/8 cloud cover',
         7.0: '7/8 cloud cover',
         8.0: '8/8 cloud cover'}

    V = {1.0: 'Bad (< 1km)',
         2.0: 'Middle (1km - 4 km)',
         3.0: 'Good (4 km - 10 km)',
         4.0: 'Excellent (> 10 km)'}

    VQ1 = {0.0: 'No wind',
           1.0: 'Fohn',
           2.0: 'Wind activity with slab formation',
           3.0: 'Strong wind activity',
           4.0: 'Wind activity without snow drift'}

    VQ2 = {0.0: 'No slab',
           1.0: 'North exposed slopes',
           2.0: 'Wind activity with slab formation',
           3.0: 'South exposed slopes',
           4.0: 'West exposed slopes',
           5.0: 'Slopes in all aspects'}

    CS = {11: 'Dry loose snow',
          12: 'Dry melt-freeze crust load-bearing ',
          13: 'Dry melt-freeze crust non-load-bearing crust',
          14: 'Dry wind crust load-bearing',
          15: 'Dry wind crust non-load-bearing crust',

          21: 'Wet loose snow',
          22: 'Wet melt-freeze crust load-bearing',
          23: 'Wet melt-freeze crust non-load-bearing crust',
          24: 'Wet wind crust load-bearing',
          25: 'Wet wind crust non-load-bearing crust'}

    S = {1.0: 'Smooth',
         2.0: 'Qavy',
         3.0: 'Concave furrows',
         4.0: 'Convex furrows',
         5.0: 'Random furrows'
         }

    B = {0.0: 'No surface hoar',
         1.0: 'New surface hoar (< 2cm)',
         2.0: 'Old surface hoar (> 2cm)',
         3.0: 'Old hoar with no evolution',
         }

    L1 = {1.0: 'Small avalanches (sluff)',
          2.0: 'Medium-size avalanches',
          3.0: 'Many medium-sized avalanches',
          4.0: 'Single large avalanches',
          5.0: 'Several large avalanches',
          6.0: 'Old classification',
          7.0: 'Old classification',
          8.0: 'Old classification',
          9.0: 'Old classification'}

    L2 = {
        1.0: 'Surface slab avalanche slab avalanches',
        2.0: 'Ground slab avalanches',
        3.0: 'Surface Loose snow avalanches',
        4.0: 'Ground Loose snow avalanches',
        5.0: 'Surface slab and loose snow avalanches',
        6.0: 'Ground slab and loose snow avalanches'
    }

    L3 = {
        1.0: 'North exposed slopes',
        2.0: 'East exposed slopes',
        3.0: 'South exposed slopes',
        4.0: 'West exposed slopes',
        5.0: 'Shade exposed slopes',
        6.0: 'Slopes exposed to sunlight',
        7.0: 'Slopes in all aspects',
        8.0: 'Slopes sheltered from the winds'
    }

    L4 = {1.0: 'Below 1000 m',
          2.0: '1000 - 1500 m',
          3.0: 'Below 1500 m',
          4.0: '1500 - 2000 m',
          5.0: 'Below 2000 m',
          6.0: '1800 - 2300 m',
          7.0: '2300 - 2800 m',
          8.0: 'Above 2800 m',
          9.0: 'Various altitudes'}

    L5 = {1.0: '07-11',
          2.0: '11-16',
          3.0: '16-20',
          4.0: 'During day (07-20)',
          5.0: 'During night (20-07)',
          6.0: 'During 24h'}

    L6 = {1.0: 'Large group of skiers',
          2.0: 'Single skier',
          3.0: 'Explosive trigger: no release',
          4.0: 'Explosive trigger: small/medium avalanches',
          5.0: 'Explosive trigger: large avalanches',
          6.0: 'Snow groomer release'}

    # Combine both dictionaries into one
    data = {"WW": WW,
            "N": N,
            "V": V,
            "VQ1": VQ1,
            "VQ2": VQ2,
            "CS": CS,
            "S": S,
            "B": B,
            "L1": L1,
            "L2": L2,
            "L3": L3,
            "L4": L4,
            "L5": L5,
            "L6": L6
            }

    return data


def write_codice_nivometeo(filepath):
    '''
    Function to write a json file containig the code classification for each 
    observed parameter.

    Parameters
    ----------
    filepath : pathlib.WindowsPath
        the filepath where to save the file.

    Returns
    -------
    None.

    '''
    data = codice_nivometeorologico()
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)


def main():
    '''
    Main function

    Returns
    -------
    None.

    '''

    data_folder = Path(
        "C:\\Users\\Christian\\OneDrive\\Desktop\\Family\\Christian\\MasterMeteoUnitn\\Corsi\\4_Tesi\\03_Dati\\")

    file = 'codice_nivometeorologico.json'
    filepath = data_folder / file

    # Write the combined dictionary to a JSON file
    write_codice_nivometeo(filepath)


if __name__ == '__main__':
    main()
