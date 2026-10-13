import json,pathlib,sys,requests
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    url='https://api.neso.energy/api/3/action/package_show'
    response=requests.get(url,params={'id':'91c0c70e-0ef5-4116-b6fa-7ad084b5e0e8'},timeout=(15,45));response.raise_for_status()
    payload=response.json();assert payload['success']
    (ROOT/'outputs/source_metadata/package_show.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    for item in payload['result']['resources']:
        print('RESOURCE',item['id'],item.get('name'),item.get('url'),flush=True)
if __name__=='__main__':main()
