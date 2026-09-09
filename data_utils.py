import os 
import datetime
import requests



def get_donki_cmes(date_start,date_end,data_type='CME',file_path='data/donki_data.json'):
    """
    Download DONKI data for the specified data type and save it as a JSON file.

    """

    data_dir = os.path.dirname(file_path)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    url_donki = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/'+data_type+'?startDate='+date_start.strftime("%Y-%m-%d")+'&endDate='+date_end.strftime("%Y-%m-%d")

    try:
        response = requests.get(url_donki, timeout=600)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx/5xx)

        # Save content as JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download DONKI data: {e}")


if __name__ == "__main__":
    get_donki_cmes(datetime.datetime(2026,1,12),datetime.datetime(2026,1,17))