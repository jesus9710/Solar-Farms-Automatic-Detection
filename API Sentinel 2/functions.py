import requests
import time

def get_credentials(text_file):
    """ Permite obtener las credenciales a partir del fichero de texto indicado
    Formato:
        user = <mi_usuario>
        password = <mi_contraseña>
    """
    creds_dict ={}
    with open('creds.txt') as file:
        for line in file.readlines():
            c_line = line.split('=')
            creds_dict.update({c_line[0].strip():c_line[1].strip()})

    class creds():
        def __init__(self, credentials):
            self.user = credentials['user']
            self.password = credentials['password']
        
    return creds(creds_dict)

creds = get_credentials('creds.txt')

def get_access_token(username: str, password: str) -> str:
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    try:
        r = requests.post(
            "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
            data=data,
        )
        r.raise_for_status()
    except Exception as e:
        raise Exception(
            f"Access token creation failed. Reponse from the server was: {r.json()}"
        )
    return r.json()["access_token"]


def download_product(access_token, Id, filename, dir = None, time_enable = False):

    """ permite descargar un producto especificando la Id
    """

    if time_enable:
        start_time = time.time()

    dir = '' if dir == None else (dir if dir[-1] == '/' else dir + '/')

    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({Id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}
    session = requests.Session()
    session.headers.update(headers)
    response = session.get(url, headers=headers, stream=True)

    with open(dir + filename + '.zip', "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
    
    if time_enable:
        print("--- %s seconds ---" % (time.time() - start_time))


def download_products(access_token, Ids, file_names = None, dir = None, time_enable = False, ind = 'numeric'):

    """ permite descargar varios productos. Para ello se necesita especificar una lista con las Ids que se desean descargar
    """

    dir = '' if dir == None else (dir if dir[-1] == '/' else dir + '/')

    if file_names == None:
        file_names = [dir + 'Product_' + str(i) if ind == 'numeric' else dir + 'Product_' + str(j) for i,j in enumerate(Ids)]

    for id, file in zip(Ids, file_names):

        download_product(access_token = access_token, Id = id, filename= file, dir=None, time_enable = time_enable)