#

PATH_STORAGE = "storage"
PATH_CLUSTER = PATH_STORAGE + "/cluster"
PATH_LEMMA = PATH_STORAGE + "/lemma"
PATH_THEME = PATH_STORAGE + "/theme"

PATH_KB = PATH_STORAGE + "/kb"

PATH_INCIDENCIAS = PATH_STORAGE + "/incidencias"

PATH_IMGS_CODO= PATH_STORAGE + "/imgs/codo"
PATH_IMGS_CLUSTER=PATH_STORAGE + "/imgs/cluster"


CSV_CONSOLIDATE = PATH_STORAGE + "/incidencias_consolidadas.csv"

PATH_DATASET = PATH_STORAGE + "/datasets"

PATH_WHITE_LIST = PATH_DATASET + "/whitelist"

PATH_BLACK_LIST = PATH_DATASET + "/blacklist"


COLLECTION_NAME_PROBLEM = 'problems'

PATH_CHROMA_DB = PATH_STORAGE + '/knowledge_base_db'

def clean_name(name):
    return name.replace(' ', '_').replace('-', '_')

